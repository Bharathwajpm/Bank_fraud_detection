"""
Root FastAPI API entrypoint.

Re-exports the production FastAPI app from app.main.
Allows both `uvicorn app.main:app` and `uvicorn api:app`.
"""

from app.main import app

__all__ = ["app"]