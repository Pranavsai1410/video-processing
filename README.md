# Video Processing FastAPI Application

This is a FastAPI application that processes video files, extracts frames, computes feature vectors, stores them in a Qdrant vector database, and provides an endpoint to search for similar frames.

## Features
- **Video Upload**: Accepts MP4 video files and extracts frames at 1-second intervals.
- **Frame Processing**: Saves frames as JPEG images and computes RGB color histogram feature vectors.
- **Vector Storage**: Stores feature vectors in Qdrant for efficient similarity search.
- **Similarity Search**: Allows querying similar frames based on an uploaded image's feature vector, returning frame images and vectors.

## Output 

## Prerequisites
- Docker and Docker Compose
- Python 3.11 (if running without Docker)
- Qdrant vector database

## Setup Instructions

### Using Docker
1. Navigate to the project directory:
   ```bash
   cd video-processing-fastapi
