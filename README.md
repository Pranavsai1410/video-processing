# Video Processing FastAPI Application

This is a FastAPI application that processes video files, extracts frames, computes feature vectors, stores them in a Qdrant vector database, and provides an endpoint to search for similar frames.

## Features
- **Video Upload**: Accepts MP4 video files and extracts frames at 1-second intervals.
- **Frame Processing**: Saves frames as JPEG images and computes RGB color histogram feature vectors.
- **Vector Storage**: Stores feature vectors in Qdrant for efficient similarity search.
- **Similarity Search**: Allows querying similar frames based on an uploaded image's feature vector, returning frame images and vectors.

## Output 
###  1. FastAPI Swagger UI (`localhost:8000/docs`)
![Swagger UI](https://github.com/Pranavsai1410/video-processing/blob/main/Screenshot%202025-06-26%20163924.png)

---

###  2. Upload Endpoint – Frame Extraction and Vector Insertion
![Upload Endpoint](https://github.com/Pranavsai1410/video-processing/blob/main/Screenshot%202025-06-26%20164027.png)

---

###  3. Output Frames Folder Structure
![Output Frames](https://github.com/Pranavsai1410/video-processing/blob/main/Screenshot%202025-06-26%20164049.png)

---

###  4. Search Endpoint – Matching Frame Results
![Search Results JSON](https://github.com/Pranavsai1410/video-processing/blob/main/Screenshot%202025-06-26%20164133.png)

---

###  5. Decoded Result Images from Search
![Decoded Results](https://github.com/Pranavsai1410/video-processing/blob/main/Screenshot%202025-06-26%20164147.png)

## Frame Search Example

This example demonstrates how the system finds visually similar frames using feature vectors (color histograms) and Qdrant.

### Input Frame (Query Image)
This is the input frame used to query the vector database:

![Input Frame](https://github.com/Pranavsai1410/video-processing/blob/main/frame_19.jpg)

---

### Output Frame (Top Search Result)
This is the most similar frame retrieved from the database:

![Output Frame](https://raw.githubusercontent.com/Pranavsai1410/video-processing/refs/heads/main/result_0.jpg)


## Prerequisites
- Docker and Docker Compose
- Python 3.11 (if running without Docker)
- Qdrant vector database

## Setup Instructions

### Using Docker
1. Navigate to the project directory:
   ```bash
   cd video-processing-fastapi
