"""
routers/admin.py — Admin-only endpoints
"""
import json
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session

from be.database import get_db
from be.models import User, Submission, RetrainJob
from be.schemas import (
    AdminStats,
    UserOut,
    UserUpdate,
    SubmissionOut,
    ReviewRequest,
    RetrainRequest,
    RetrainJobOut,
)
from be.auth import get_current_user, require_admin

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=AdminStats)
def get_stats(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    total_users = db.query(User).count()
    total_subs = db.query(Submission).count()
    pending = db.query(Submission).filter(Submission.status == "pending").count()
    approved = db.query(Submission).filter(Submission.status == "approved").count()
    rejected = db.query(Submission).filter(Submission.status == "rejected").count()
    total_jobs = db.query(RetrainJob).count()

    last_job = db.query(RetrainJob).filter(RetrainJob.status == "done").order_by(RetrainJob.finished_at.desc()).first()

    return AdminStats(
        total_users=total_users,
        total_submissions=total_subs,
        pending_count=pending,
        approved_count=approved,
        rejected_count=rejected,
        total_retrain_jobs=total_jobs,
        last_retrain_at=last_job.finished_at if last_job else None,
    )


# ── Users ─────────────────────────────────────────────────────────────────────

@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    return db.query(User).order_by(User.created_at.desc()).all()


@router.put("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: int,
    req: UserUpdate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User không tìm thấy")
    if user.id == current_admin.id:
        raise HTTPException(status_code=400, detail="Không thể sửa tài khoản của chính mình")

    if req.role is not None:
        if req.role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="Role phải là 'user' hoặc 'admin'")
        user.role = req.role
    if req.is_active is not None:
        user.is_active = req.is_active

    db.commit()
    db.refresh(user)
    return user


# ── Submissions ───────────────────────────────────────────────────────────────

@router.get("/submissions", response_model=list[SubmissionOut])
def list_submissions(
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    q = db.query(Submission).order_by(Submission.created_at.desc())
    if status:
        q = q.filter(Submission.status == status)
    return q.all()


@router.get("/submissions/{submission_id}", response_model=SubmissionOut)
def get_submission(
    submission_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    s = db.query(Submission).filter(Submission.id == submission_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Submission không tìm thấy")
    return s


@router.put("/submissions/{submission_id}/approve", response_model=SubmissionOut)
def approve_submission(
    submission_id: int,
    req: ReviewRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    s = db.query(Submission).filter(Submission.id == submission_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Submission không tìm thấy")
    if s.status != "pending":
        raise HTTPException(status_code=400, detail="Chỉ có thể duyệt submission đang pending")

    s.status = "approved"
    s.admin_note = req.admin_note
    s.reviewed_by = current_admin.id
    s.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(s)
    return s


@router.put("/submissions/{submission_id}/reject", response_model=SubmissionOut)
def reject_submission(
    submission_id: int,
    req: ReviewRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    s = db.query(Submission).filter(Submission.id == submission_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Submission không tìm thấy")
    if s.status != "pending":
        raise HTTPException(status_code=400, detail="Chỉ có thể từ chối submission đang pending")

    s.status = "rejected"
    s.admin_note = req.admin_note
    s.reviewed_by = current_admin.id
    s.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(s)
    return s


# ── Retrain ───────────────────────────────────────────────────────────────────

@router.post("/retrain/start", response_model=RetrainJobOut, status_code=201)
def start_retrain(
    req: RetrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    # Validate submissions are all approved
    subs = db.query(Submission).filter(Submission.id.in_(req.submission_ids)).all()
    if len(subs) != len(req.submission_ids):
        raise HTTPException(status_code=404, detail="Một số submission không tồn tại")
    not_approved = [s for s in subs if s.status != "approved"]
    if not_approved:
        raise HTTPException(
            status_code=400,
            detail=f"{len(not_approved)} submission chưa được duyệt. Chỉ có thể retrain với tin đã approve.",
        )

    # Check no job currently running
    running = db.query(RetrainJob).filter(RetrainJob.status.in_(["queued", "running"])).first()
    if running:
        raise HTTPException(status_code=409, detail="Đang có một job retrain đang chạy. Chờ hoàn thành rồi thử lại.")

    job = RetrainJob(
        triggered_by=current_admin.id,
        submission_ids=json.dumps(req.submission_ids),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Run in background
    from be.ml.retrainer import run_retrain_job
    background_tasks.add_task(run_retrain_job, job.id)

    return job


@router.get("/retrain/jobs", response_model=list[RetrainJobOut])
def list_retrain_jobs(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    return db.query(RetrainJob).order_by(RetrainJob.created_at.desc()).all()


@router.get("/retrain/jobs/{job_id}", response_model=RetrainJobOut)
def get_retrain_job(
    job_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    job = db.query(RetrainJob).filter(RetrainJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job không tìm thấy")
    return job
