from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import os
import shutil
import base64
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
import logging
import io

app = FastAPI(title="Video Processing API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
COLLECTION_NAME = "video_frames"

# Output directory for frames
OUTPUT_DIR = "output_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def initialize_qdrant():
    """Initialize Qdrant collection if it doesn't exist."""
    try:
        collections = qdrant_client.get_collections().collections
        logger.info("Existing collections: %s", [c.name for c in collections])
        if not any(collection.name == COLLECTION_NAME for collection in collections):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=192, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", COLLECTION_NAME)
        else:
            logger.info("Collection %s already exists", COLLECTION_NAME)
        # Verify collection
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        logger.info("Collection info: %s", collection_info)
    except Exception as e:
        logger.error("Failed to initialize Qdrant: %s", str(e))
        raise

initialize_qdrant()

def compute_color_histogram(image: np.ndarray) -> np.ndarray:
    """Compute RGB color histogram for an image."""
    hist_r = cv2.calcHist([image], [0], None, [64], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [64], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [64], [0, 256])
    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    hist = hist / hist.sum()  # Normalize
    return hist

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    logger.info("Received upload request for file: %s", file.filename)
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 videos are supported")
    try:
        video_id = str(uuid.uuid4())
        frame_dir = os.path.join("output_frames", video_id)
        os.makedirs(frame_dir, exist_ok=True)
        video_path = os.path.join(frame_dir, file.filename)
        with open(video_path, "wb") as f:
            f.write(await file.read())
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")
        frame_count = 0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
                frame_path = os.path.join(frame_dir, f"frame_{len(frames)}.jpg")
                cv2.imwrite(frame_path, frame)
                hist = compute_color_histogram(frame)
                frames.append({"path": frame_path, "histogram": hist})
            frame_count += 1
        cap.release()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=frame["histogram"].tolist(),
                payload={
                    "video_id": video_id,
                    "frame_id": str(uuid.uuid4()),
                    "path": frame["path"],
                },
            )
            for frame in frames
        ]
        logger.info("Prepared %d points for Qdrant", len(points))
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        logger.info("Stored %d frames in Qdrant for video %s", len(points), video_id)
        return {
            "status": "success",
            "video_id": video_id,
            "frame_count": len(frames),
            "frame_paths": [frame["path"] for frame in frames],
        }
    except Exception as e:
        logger.error("Upload error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_similar_frames(file: UploadFile = File(...)):
    logger.info("Received search request for file: %s", file.filename)
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only PNG or JPEG images are supported")
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)
        hist = compute_color_histogram(img_array)
        logger.info("Computed histogram (first 10 values): %s", hist[:10])
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=hist.tolist(),
            limit=5,
            with_vectors=True
        )
        logger.info("Search returned %d results", len(search_result))
        results = []
        for hit in search_result:
            logger.info("Hit: ID=%s, Score=%f, Payload=%s, Vector=%s", 
                        hit.id, hit.score, hit.payload, hit.vector)
            frame_path = hit.payload.get("path")
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                    results.append({
                        "frame_id": hit.payload["frame_id"],
                        "video_id": hit.payload["video_id"],
                        "score": float(hit.score),
                        "image_data": f"data:image/jpeg;base64,{image_data}",
                        "feature_vector": hit.vector if hit.vector is not None else []
                    })
            else:
                logger.warning("Frame path not found: %s", frame_path)
        return {"results": results}
    except Exception as e:
        logger.error("Search error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))