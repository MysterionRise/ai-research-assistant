"""Celery application configuration."""

from celery import Celery

from aria.config.settings import settings

# Create Celery app
celery_app = Celery(
    "aria",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Configure Celery
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task tracking
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,  # 50 minutes soft limit
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    # Result settings
    result_expires=86400,  # 24 hours
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["aria.worker.tasks"])
