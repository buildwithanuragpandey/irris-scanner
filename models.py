import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, Enum, LargeBinary, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class EnrolledIris(Base):
    __tablename__ = "enrolled_iris"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    voter_id_hash = Column(String(64), unique=True, index=True, nullable=False)
    iris_code_encrypted = Column(LargeBinary, nullable=False)
    mask_encrypted = Column(LargeBinary, nullable=False)
    cnn_embedding_encrypted = Column(LargeBinary, nullable=False)
    eye_side = Column(Enum('left', 'right', name='eye_side_enum'), nullable=False)
    quality_score = Column(Float)
    enrolled_at = Column(DateTime, default=datetime.utcnow)
    encryption_key_id = Column(String, default="v1")
    is_revoked = Column(Boolean, default=False)

class MatchAuditLog(Base):
    __tablename__ = "match_audit_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    voter_id_hash = Column(String(64), index=True)
    attempt_at = Column(DateTime, default=datetime.utcnow)
    hamming_distance = Column(Float)
    cosine_similarity = Column(Float)
    decision = Column(Enum('match', 'no_match', 'poor_quality', 'liveness_fail', name='decision_enum'))
    attempt_number = Column(Integer)
    device_id_hash = Column(String(64))
    ip_hash = Column(String(64))
