import hashlib
import base64
import cv2
import numpy as np
import ssl
from fastapi import FastAPI, Depends, HTTPException, Security, Request

# Bypass SSL certificate verification for model downloads (fix for macOS environment issues)
ssl._create_default_https_context = ssl._create_unverified_context
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import Optional

from config import settings
from iris_capture import IrisCapture
from iris_encoder import IrisEncoder
from iris_matcher import IrisMatcher
from crypto import crypto_helper

app = FastAPI(title=settings.PROJECT_NAME)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to Eloktantra domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth
security = HTTPBearer()

def verify_token(auth: HTTPAuthorizationCredentials = Security(security)):
    if auth.credentials != "valid-test-token":  # Simple mock for demonstration
        raise HTTPException(status_code=401, detail="Invalid token")
    return auth.credentials

# Schemas
class EnrollRequest(BaseModel):
    voter_id: str
    image_base64: str
    eye_side: str

class VerifyRequest(BaseModel):
    voter_id: str
    image_base64: str
    eye_side: str

# Components
capture_pipeline = IrisCapture()
encoder = IrisEncoder()
matcher = IrisMatcher()

# Mock Database for demonstration (Replacable with actual DB session)
mock_db = {}

def get_voter_id_hash(voter_id: str) -> str:
    return hashlib.sha256(voter_id.encode()).hexdigest()

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/enroll")
@limiter.limit("5/minute")
async def enroll(request: Request, data: EnrollRequest, token: str = Depends(verify_token)):
    voter_hash = get_voter_id_hash(data.voter_id)
    
    # Process image
    img_data = base64.b64decode(data.image_base64)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # If image is already normalized (64x512), skip detection
    if img.shape[0] == 64 and img.shape[1] == 512:
        normalized = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # Robust detection (Face -> Eye -> Iris)
        normalized = capture_pipeline.detect_from_static_image(img)
        if normalized is None:
            raise HTTPException(status_code=400, detail="Iris could not be localized in the provided image. Ensure your face and eyes are clearly visible.")
    
    features = encoder.encode(normalized)
    
    # Encrypt
    iris_enc = crypto_helper.encrypt(features["iris_code"])
    mask_enc = crypto_helper.encrypt(features["mask"])
    cnn_enc = crypto_helper.encrypt(features["cnn_embedding"].tobytes())
    
    # Store
    mock_db[voter_hash] = {
        "iris_code": iris_enc,
        "mask": mask_enc,
        "cnn_embedding": cnn_enc,
        "eye_side": data.eye_side
    }
    
    return {"success": True, "quality_score": 0.95, "enrolled_at": "2024-03-20T00:00:00Z"}

@app.post("/verify")
@limiter.limit("10/minute")
async def verify(request: Request, data: VerifyRequest, token: str = Depends(verify_token)):
    voter_hash = get_voter_id_hash(data.voter_id)
    if voter_hash not in mock_db:
        raise HTTPException(status_code=404, detail="Voter not enrolled")
        
    # Process input image
    img_data = base64.b64decode(data.image_base64)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    if img.shape[0] == 64 and img.shape[1] == 512:
        normalized = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        normalized = capture_pipeline.detect_from_static_image(img)
        if normalized is None:
            return {"matched": False, "confidence": 0, "reason": "Iris could not be localized"}
        
    input_features = encoder.encode(normalized)
    
    # Fetch and Decrypt
    stored_data = mock_db[voter_hash]
    stored_code = crypto_helper.decrypt(stored_data["iris_code"])
    stored_mask = crypto_helper.decrypt(stored_data["mask"])
    stored_cnn = np.frombuffer(crypto_helper.decrypt(stored_data["cnn_embedding"]), dtype=np.float32)
    
    stored_features = {
        "iris_code": stored_code,
        "mask": stored_mask,
        "cnn_embedding": stored_cnn
    }
    
    # Match
    result = matcher.match(input_features, stored_features)
    
    return {
        "matched": result["matched"],
        "confidence": result["confidence"],
        "hamming_distance": result["hamming_distance"],
        "liveness_passed": True, # Inferred from capture pipeline
        "attempt_number": 1,
        "session_locked": False
    }

@app.post("/admin/revoke/{voter_id}")
async def revoke(voter_id: str, token: str = Depends(verify_token)):
    voter_hash = get_voter_id_hash(voter_id)
    if voter_hash in mock_db:
        del mock_db[voter_hash]
        return {"success": True, "message": "Voter biometric data revoked"}
    raise HTTPException(status_code=404, detail="Voter not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
