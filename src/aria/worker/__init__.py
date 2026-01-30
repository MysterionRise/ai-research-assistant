"""Celery worker for background task processing."""

from aria.worker.celery_app import celery_app

__all__ = ["celery_app"]
