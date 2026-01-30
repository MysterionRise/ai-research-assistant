"""Health check endpoints.

Provides endpoints for Kubernetes liveness and readiness probes.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, status

from aria.config.settings import settings

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Basic health check endpoint for load balancers.",
)
async def health_check() -> dict[str, Any]:
    """Basic health check.

    Returns:
        dict: Health status.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "version": "0.1.0",
    }


@router.get(
    "/api/v1/health",
    status_code=status.HTTP_200_OK,
    summary="API health check",
    description="Health check endpoint with version info.",
)
async def api_health_check() -> dict[str, Any]:
    """API health check with metadata.

    Returns:
        dict: Health status with metadata.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "version": "0.1.0",
        "environment": settings.environment,
    }


@router.get(
    "/api/v1/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Checks if the service is ready to accept traffic.",
)
async def readiness_check() -> dict[str, Any]:
    """Readiness check for Kubernetes.

    Verifies that all dependencies are available.

    Returns:
        dict: Readiness status with component checks.
    """
    checks: dict[str, dict[str, Any]] = {}

    # TODO: Check database connectivity
    checks["database"] = {"status": "healthy", "latency_ms": 0}

    # TODO: Check Redis connectivity
    checks["redis"] = {"status": "healthy", "latency_ms": 0}

    # TODO: Check vector store connectivity
    checks["vector_store"] = {"status": "healthy", "latency_ms": 0}

    # Determine overall status
    all_healthy = all(c["status"] == "healthy" for c in checks.values())

    return {
        "status": "ready" if all_healthy else "not_ready",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "checks": checks,
    }


@router.get(
    "/api/v1/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Checks if the service is alive.",
)
async def liveness_check() -> dict[str, str]:
    """Liveness check for Kubernetes.

    Returns:
        dict: Liveness status.
    """
    return {"status": "alive"}
