"""
database.py — SQLite + SQLAlchemy setup
"""
import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DB_PATH = os.path.join(os.path.dirname(__file__), "vifn.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables and seed default admin + demo users."""
    from be.models import User, Submission  # import here to avoid circular
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Seed admin if not exists
        if not db.query(User).filter(User.username == "admin").first():
            from be.auth import get_password_hash
            admin = User(
                username="admin",
                email="admin@vifn.vn",
                hashed_password=get_password_hash("admin123"),
                role="admin",
                is_active=True,
            )
            db.add(admin)

        # Seed demo user
        if not db.query(User).filter(User.username == "demo").first():
            from be.auth import get_password_hash
            demo = User(
                username="demo",
                email="demo@vifn.vn",
                hashed_password=get_password_hash("demo123"),
                role="user",
                is_active=True,
            )
            db.add(demo)

        db.commit()
    finally:
        db.close()
