"""
routers/news.py — User news endpoints (predict + submit + history)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from be.database import get_db
from be.models import User, Submission
from be.schemas import PredictRequest, SubmitRequest, SubmissionOut
from be.auth import get_current_user

router = APIRouter(prefix="/api/news", tags=["news"])


@router.post("/predict")
def predict_news(req: PredictRequest, current_user: User = Depends(get_current_user)):
    """Run PhoBERT + Retrieval prediction on provided text."""
    from be.ml.predictor import run_prediction
    try:
        result = run_prediction(
            text=req.text,
            max_length=req.max_length,
            top_k_bm25=req.top_k_bm25,
            top_k_web=req.top_k_web,
            enable_web=req.enable_web,
            crawl_web=req.crawl_web,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi chạy mô hình: {str(e)}")


@router.post("/submit", response_model=SubmissionOut, status_code=201)
def submit_news(
    req: SubmitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit a verified news article for admin review."""
    submission = Submission(
        user_id=current_user.id,
        title=req.title,
        content=req.content,
        user_label=req.user_label,
        source_url=req.source_url,
        status="pending",
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)
    return submission


@router.get("/my", response_model=list[SubmissionOut])
def my_submissions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all submissions made by the current user."""
    return (
        db.query(Submission)
        .filter(Submission.user_id == current_user.id)
        .order_by(Submission.created_at.desc())
        .all()
    )
