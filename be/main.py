"""
main.py — FastAPI application entry point
Run: python -m uvicorn be.main:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from be.database import init_db
from be.routers import auth, news, admin

app = FastAPI(
    title="ViFN Fake News Platform API",
    description="Vietnamese Fake News Detection — PhoBERT + RAG",
    version="2.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(news.router)
app.include_router(admin.router)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {"message": "ViFN API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
