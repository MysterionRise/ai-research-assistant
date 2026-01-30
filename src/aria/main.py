"""ARIA Application Entry Point.

Run with: python -m aria.main
Or: uvicorn aria.main:app --reload
"""

import uvicorn

from aria.api.app import app
from aria.config.settings import settings


def main() -> None:
    """Run the ARIA application."""
    uvicorn.run(
        "aria.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

# Export app for uvicorn
__all__ = ["app", "main"]
