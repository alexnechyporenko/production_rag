"""Ingestion package: loader + ingestion service. Chunkers live in `app.chunking`."""

from app.ingestion.loader import DocumentLoader
from app.ingestion.service import IngestionService

__all__ = ["DocumentLoader", "IngestionService"]
