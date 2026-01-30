"""Celery tasks for ARIA."""

from aria.worker.tasks.ingestion import ingest_document

__all__ = ["ingest_document"]
