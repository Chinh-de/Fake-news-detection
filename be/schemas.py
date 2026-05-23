"""
schemas.py — Pydantic request/response schemas
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


# ── Auth ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None


# ── News / Submissions ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    max_length: int = 256
    top_k_bm25: int = 4
    top_k_web: int = 3
    enable_web: bool = True
    crawl_web: bool = False   # default off for speed


class PredictResponse(BaseModel):
    label: str
    confidence: float
    prob_real: float
    prob_fake: float
    explanation: dict
    bm25_evidences: list
    web_evidences: list
    corpus_name: str
    corpus_total: int


class SubmitRequest(BaseModel):
    title: str = Field(..., min_length=5, max_length=300)
    content: str = Field(..., min_length=20)
    user_label: str = Field(..., pattern="^(REAL|FAKE)$")
    source_url: Optional[str] = None


class SubmissionOut(BaseModel):
    id: int
    title: str
    content: str
    user_label: str
    source_url: Optional[str]
    status: str
    admin_note: Optional[str]
    created_at: datetime
    reviewed_at: Optional[datetime]
    author: Optional[UserOut] = None
    reviewer: Optional[UserOut] = None

    class Config:
        from_attributes = True


# ── Admin ─────────────────────────────────────────────────────────────────────

class AdminStats(BaseModel):
    total_users: int
    total_submissions: int
    pending_count: int
    approved_count: int
    rejected_count: int
    total_retrain_jobs: int
    last_retrain_at: Optional[datetime]


class ReviewRequest(BaseModel):
    admin_note: Optional[str] = None


class RetrainRequest(BaseModel):
    submission_ids: List[int]


class RetrainJobOut(BaseModel):
    id: int
    submission_ids: str
    status: str
    accuracy_before: Optional[float]
    accuracy_after: Optional[float]
    log_text: Optional[str]
    created_at: datetime
    finished_at: Optional[datetime]
    triggered_by_user: Optional[UserOut] = None

    class Config:
        from_attributes = True
