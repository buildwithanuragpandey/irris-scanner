# Eloktantra Iris Recognition Module

Production-ready iris recognition and matching microservice for the Eloktantra secure e-voting platform.

## Features
- **Real-time Capture**: OpenCV + dlib facial landmarking for eye localization.
- **Iris Segmentation**: Daugman-style integro-differential operator implemented via Hough Circles.
- **Daugman Normalization**: Rubber Sheet model mapping iris to a 64x512 strip.
- **Liveness Detection**: Multi-factor (Blink detection, EAR tracking, texture variance).
- **Two-Stage Matching**:
  1. **Hamming Distance**: Fast pre-filter with rotational bit-shifting compensation.
  2. **CNN Embeddings**: Precision matching using MobileNetV2 features + Cosine Similarity.
- **Security**: AES-256-GCM encryption at rest, SHA-256 voter ID hashing, Rate limiting (slowapi).

## Setup

### Local Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download dlib landmarks:
   ```bash
   mkdir models
   # Download shape_predictor_68_face_landmarks.dat into models/
   ```
3. Run the API:
   ```bash
   uvicorn api:app --reload
   ```

### Docker Deployment
```bash
docker build -t eloktantra-iris .
docker run -p 8000:8000 eloktantra-iris
```

## API Documentation

### POST /enroll
Enroll a new voter's iris.
```json
{
  "voter_id": "Voter123",
  "image_base64": "...",
  "eye_side": "left"
}
```

### POST /verify
Verify a voter's iris against enrolled data.
```json
{
  "voter_id": "Voter123",
  "image_base64": "...",
  "eye_side": "left"
}
```

### GET /health
System health check.

## Testing
Run unit tests:
```bash
pytest tests/
```
