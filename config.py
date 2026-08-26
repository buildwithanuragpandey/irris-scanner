import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    PROJECT_NAME: str = "Eloktantra Iris Recognition Module"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-key-for-jwt")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Biometric Thresholds
    SHARPNESS_THRESHOLD: float = 100.0  # Laplacian variance
    IRIS_VISIBILITY_THRESHOLD: float = 0.70
    PUPIL_IRIS_RATIO_MIN: float = 0.2
    PUPIL_IRIS_RATIO_MAX: float = 0.7
    CONCENTRIC_TOLERANCE: int = 15  # pixels
    
    # Matching Thresholds
    HAMMING_THRESHOLD: float = 0.32
    COSINE_SIM_THRESHOLD: float = 0.92
    
    # Security
    MAX_MATCH_ATTEMPTS: int = 3
    SESSION_LOCK_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/eloktantra_iris")
    
    # Paths
    DLIB_LANDMARK_PATH: str = os.getenv("DLIB_LANDMARK_PATH", "models/shape_predictor_68_face_landmarks.dat")
    
    class Config:
        case_sensitive = True

settings = Settings()
