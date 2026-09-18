"""Structured error taxonomy for the RAG system.

Every recoverable failure raises a `RAGError` subclass. The FastAPI layer
maps these to HTTP responses with a stable JSON shape, so clients can
dispatch on `code` rather than parsing prose.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Stable, machine-readable error codes."""

    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DOCUMENT_PARSE_ERROR = "DOCUMENT_PARSE_ERROR"
    DOCUMENT_TOO_LARGE = "DOCUMENT_TOO_LARGE"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    LEXICAL_SEARCH_ERROR = "LEXICAL_SEARCH_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    RERANKING_ERROR = "RERANKING_ERROR"
    CONTEXT_LIMIT_ERROR = "CONTEXT_LIMIT_ERROR"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_OUTPUT_ERROR = "LLM_OUTPUT_ERROR"
    INVALID_CITATION = "INVALID_CITATION"
    EVALUATION_ERROR = "EVALUATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    UNKNOWN = "UNKNOWN"


class RAGError(Exception):
    """Base class for all RAG-domain errors.

    Attributes:
        code: stable ErrorCode.
        message: human-readable detail (will be sanitised before logging).
        details: structured, non-sensitive context (never secrets).
        cause: optional underlying exception.
    """

    code: ErrorCode = ErrorCode.UNKNOWN
    http_status: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


# --- Configuration ----------------------------------------------------------


class ConfigurationError(RAGError):
    code = ErrorCode.CONFIGURATION_ERROR
    http_status = 500


# --- Ingestion --------------------------------------------------------------


class DocumentParseError(RAGError):
    code = ErrorCode.DOCUMENT_PARSE_ERROR
    http_status = 422


class DocumentTooLargeError(RAGError):
    code = ErrorCode.DOCUMENT_TOO_LARGE
    http_status = 413


# --- Storage / retrieval ----------------------------------------------------


class EmbeddingError(RAGError):
    code = ErrorCode.EMBEDDING_ERROR
    http_status = 502


class VectorStoreError(RAGError):
    code = ErrorCode.VECTOR_STORE_ERROR
    http_status = 500


class LexicalSearchError(RAGError):
    code = ErrorCode.LEXICAL_SEARCH_ERROR
    http_status = 500


class RetrievalError(RAGError):
    code = ErrorCode.RETRIEVAL_ERROR
    http_status = 500


class RerankingError(RAGError):
    code = ErrorCode.RERANKING_ERROR
    http_status = 500


class ContextLimitError(RAGError):
    code = ErrorCode.CONTEXT_LIMIT_ERROR
    http_status = 413


# --- Generation -------------------------------------------------------------


class LLMProviderError(RAGError):
    code = ErrorCode.LLM_PROVIDER_ERROR
    http_status = 502


class LLMTimeoutError(RAGError):
    code = ErrorCode.LLM_TIMEOUT
    http_status = 504


class LLMOutputError(RAGError):
    code = ErrorCode.LLM_OUTPUT_ERROR
    http_status = 502


class InvalidCitationError(RAGError):
    code = ErrorCode.INVALID_CITATION
    http_status = 500


# --- Evaluation -------------------------------------------------------------


class EvaluationError(RAGError):
    code = ErrorCode.EVALUATION_ERROR
    http_status = 500


# --- Validation / control ---------------------------------------------------


class ValidationError(RAGError):
    code = ErrorCode.VALIDATION_ERROR
    http_status = 422


class NotFoundError(RAGError):
    code = ErrorCode.NOT_FOUND
    http_status = 404


class RateLimitedError(RAGError):
    code = ErrorCode.RATE_LIMITED
    http_status = 429


__all__ = [
    "Code",
    "ConfigurationError",
    "ContextLimitError",
    "DocumentParseError",
    "DocumentTooLargeError",
    "EmbeddingError",
    "ErrorCode",
    "EvaluationError",
    "InvalidCitationError",
    "LLMOutputError",
    "LLMProviderError",
    "LLMTimeoutError",
    "LexicalSearchError",
    "NotFoundError",
    "RAGError",
    "RateLimitedError",
    "RerankingError",
    "RetrievalError",
    "ValidationError",
    "VectorStoreError",
]


# Convenience alias.
Code = ErrorCode
