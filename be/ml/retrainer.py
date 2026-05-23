"""
ml/retrainer.py — Mock fine-tune job (simulate training for demo)
Replace with real LoRA fine-tuning in production.
"""
import json
import time
import random
import asyncio
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session

from be.database import SessionLocal
from be.models import RetrainJob, Submission


async def run_retrain_job(job_id: int):
    """
    Mock retrain: simulates fine-tuning process with progress log.
    In production: replace with Hugging Face Trainer + LoRA.
    """
    db = SessionLocal()
    try:
        job = db.query(RetrainJob).filter(RetrainJob.id == job_id).first()
        if not job:
            return

        # Mark as running
        job.status = "running"
        job.accuracy_before = round(random.uniform(0.82, 0.88), 4)
        db.commit()

        # Get submission IDs
        submission_ids = json.loads(job.submission_ids)
        submissions = db.query(Submission).filter(Submission.id.in_(submission_ids)).all()

        logs = []
        logs.append(f"[{datetime.utcnow().isoformat()}] 🚀 Bắt đầu retrain với {len(submissions)} mẫu mới")
        logs.append(f"[{datetime.utcnow().isoformat()}] 📊 Accuracy trước retrain: {job.accuracy_before:.2%}")
        logs.append(f"[{datetime.utcnow().isoformat()}] 🔧 Chuẩn bị dữ liệu...")

        # Simulate epochs
        for epoch in range(1, 4):
            await asyncio.sleep(2)  # simulate training time
            loss = round(0.45 - epoch * 0.08 + random.uniform(-0.02, 0.02), 4)
            acc = round(job.accuracy_before + epoch * 0.02 + random.uniform(0, 0.01), 4)
            logs.append(
                f"[{datetime.utcnow().isoformat()}] Epoch {epoch}/3 — loss: {loss:.4f} — val_acc: {acc:.2%}"
            )

        await asyncio.sleep(1)
        job.accuracy_after = round(job.accuracy_before + random.uniform(0.01, 0.04), 4)
        logs.append(f"[{datetime.utcnow().isoformat()}] ✅ Hoàn thành! Accuracy sau retrain: {job.accuracy_after:.2%}")
        logs.append(f"[{datetime.utcnow().isoformat()}] 💾 Model đã được lưu vào ./model/")

        job.status = "done"
        job.log_text = "\n".join(logs)
        job.finished_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        if job:
            job.status = "failed"
            job.log_text = f"ERROR: {str(e)}"
            db.commit()
    finally:
        db.close()
