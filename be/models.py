"""
models.py — SQLAlchemy ORM models
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import relationship
from be.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(10), default="user")   # "user" | "admin"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    submissions = relationship("Submission", back_populates="author", foreign_keys="Submission.user_id")
    reviewed = relationship("Submission", back_populates="reviewer", foreign_keys="Submission.reviewed_by")
    retrain_jobs = relationship("RetrainJob", back_populates="triggered_by_user")


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(300), nullable=False)
    content = Column(Text, nullable=False)
    user_label = Column(String(10), nullable=False)   # "REAL" | "FAKE"
    source_url = Column(String(500), nullable=True)
    status = Column(String(10), default="pending")    # "pending" | "approved" | "rejected"
    admin_note = Column(Text, nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    author = relationship("User", back_populates="submissions", foreign_keys=[user_id])
    reviewer = relationship("User", back_populates="reviewed", foreign_keys=[reviewed_by])


class RetrainJob(Base):
    __tablename__ = "retrain_jobs"

    id = Column(Integer, primary_key=True, index=True)
    triggered_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    submission_ids = Column(Text, nullable=False)      # JSON string: "[1, 2, 3]"
    status = Column(String(10), default="queued")     # "queued" | "running" | "done" | "failed"
    accuracy_before = Column(Float, nullable=True)
    accuracy_after = Column(Float, nullable=True)
    log_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)

    triggered_by_user = relationship("User", back_populates="retrain_jobs")
