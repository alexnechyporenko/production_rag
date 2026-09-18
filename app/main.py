"""FastAPI application factory + application lifecycle.

`create_app()` builds the FastAPI instance, wires up the orchestrator, and
registers exception handlers that translate `RAGError` subclasses into the
stable `ErrorResponse` JSON shape.

The orchestrator is held as a module-level singleton so individual request
handlers can grab it via `get_app_orchestrator()` without threading it
through every dependency.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Make the package importable when run via `python -m app.main` or uvicorn.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.domain import DocumentTooLargeError, NotFoundError, RAGError, ValidationError
from app.observability import configure_logging, get_logger, new_request_id
from app.pipeline import RAGOrchestrator

_logger = get_logger("app.main")
_orchestrator: RAGOrchestrator | None = None


def get_app_orchestrator() -> RAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator


def reset_app_orchestrator() -> None:
    """Test helper — drops the singleton so the next call rebuilds it."""
    global _orchestrator
    _orchestrator = None


def create_app(settings: Any = None) -> FastAPI:
    """Construct the FastAPI application."""
    s = settings or get_settings()
    configure_logging(s)
    app = FastAPI(
        title="Production RAG",
        version="1.0.0",
        description=(
            "Production-oriented Retrieval-Augmented Generation system: hybrid retrieval, "
            "reranking, grounded generation, citations, evaluation, cost tracking, "
            "observability."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Exception handlers -------------------------------------------------
    @app.exception_handler(RAGError)
    async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
        rid = new_request_id()
        _logger.warning(
            "rag_error",
            code=exc.code.value,
            message=exc.message,
            request_id=rid,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "code": exc.code.value,
                "message": exc.message,
                "details": exc.details,
                "request_id": rid,
            },
        )

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        return await rag_error_handler(request, exc)

    @app.exception_handler(ValidationError)
    async def validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
        return await rag_error_handler(request, exc)

    @app.exception_handler(DocumentTooLargeError)
    async def too_large_handler(request: Request, exc: DocumentTooLargeError) -> JSONResponse:
        return await rag_error_handler(request, exc)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = new_request_id()
        _logger.exception("unhandled_error", request_id=rid, path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "code": "UNKNOWN",
                "message": "Internal server error",
                "details": {"exception": type(exc).__name__},
                "request_id": rid,
            },
        )

    # --- Middleware: request_id + access log -------------------------------
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        rid = new_request_id()
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    app.include_router(router)
    return app


# ASGI entrypoint used by uvicorn.
app = create_app()


def main() -> None:  # pragma: no cover
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "app.main:app",
        host=s.api_host,
        port=s.api_port,
        log_level=s.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
