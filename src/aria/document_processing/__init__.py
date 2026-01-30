"""Document processing pipeline for ARIA.

Provides PDF parsing, metadata extraction, and document processing capabilities.
"""

from aria.document_processing.parsers.pdf import PDFParser
from aria.document_processing.pipeline import DocumentProcessingPipeline

__all__ = [
    "DocumentProcessingPipeline",
    "PDFParser",
]
