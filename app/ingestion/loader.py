"""Document loaders.

The loader turns a *raw* input (text, bytes, file path, URL) into a normalized
`Document` whose `.content` is plain UTF-8 text. Format-specific parsing lives
here so the rest of the pipeline never needs to know about PDF, HTML, etc.

Supported source formats out of the box:
  - text/plain
  - text/markdown
  - application/json
  - text/html (very small stripper)
  - application/pdf (best-effort via pypdf if installed; otherwise DOCUMENT_PARSE_ERROR)

Each loader is small, deterministic and side-effect free.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import get_settings
from app.domain import (
    Document,
    DocumentMetadata,
    DocumentParseError,
    DocumentSource,
    DocumentTooLargeError,
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_WS_RE = re.compile(r"\s+")
_WHITESPACE_RE = re.compile(r"[ \t]+")


def _normalize(text: str) -> str:
    """Collapse runs of whitespace, preserve paragraph breaks."""
    if not text:
        return ""
    # Normalize tabs/spaces but keep newlines; strip each line.
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    # Drop empty leading/trailing lines but keep interior blank lines as paragraph separators.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


class DocumentLoader:
    """Loader with content-type dispatch and size enforcement."""

    def __init__(self, max_bytes: int | None = None) -> None:
        self.max_bytes = max_bytes if max_bytes is not None else get_settings().max_document_bytes

    # --- Public API --------------------------------------------------------
    def load_text(
        self,
        text: str,
        *,
        source: DocumentSource = DocumentSource.api,
        content_type: str = "text/plain",
        title: str | None = None,
        **metadata: Any,
    ) -> Document:
        """Load from a Python string."""
        raw = text.encode("utf-8")
        self._check_size(len(raw))
        parsed = self._parse(raw, content_type)
        return Document(
            content=_normalize(parsed),
            metadata=DocumentMetadata(
                source=source,
                content_type=content_type,
                title=title,
                extra=dict(metadata),
            ),
        )

    def load_bytes(
        self,
        data: bytes,
        *,
        content_type: str = "text/plain",
        source: DocumentSource = DocumentSource.api,
        title: str | None = None,
        **metadata: Any,
    ) -> Document:
        """Load from raw bytes with an explicit content type."""
        self._check_size(len(data))
        parsed = self._parse(data, content_type)
        return Document(
            content=_normalize(parsed),
            metadata=DocumentMetadata(
                source=source,
                content_type=content_type,
                title=title,
                extra=dict(metadata),
            ),
        )

    def load_file(self, path: str | Path, **metadata: Any) -> Document:
        """Load from a filesystem path. Content type is inferred from extension."""
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise DocumentParseError(f"File not found: {p}")
        self._check_path_safe(p)
        raw = p.read_bytes()
        self._check_size(len(raw))
        ct = _guess_content_type(p)
        parsed = self._parse(raw, ct)
        return Document(
            content=_normalize(parsed),
            metadata=DocumentMetadata(
                source=DocumentSource.file,
                content_type=ct,
                title=p.stem,
                extra={"filename": p.name, **metadata},
            ),
        )

    # --- Internals ---------------------------------------------------------
    def _check_size(self, n: int) -> None:
        if n > self.max_bytes:
            raise DocumentTooLargeError(
                f"Document is {n} bytes; limit is {self.max_bytes}",
                details={"bytes": n, "limit": self.max_bytes},
            )

    def _check_path_safe(self, p: Path) -> None:
        """Reject path traversal attacks (`..` segments) and disallowed roots.

        Paths inside the current working directory or system temp dirs are
        allowed. Absolute paths elsewhere are rejected with a structured error
        so an attacker cannot read arbitrary files via the API.
        """
        import tempfile

        if ".." in p.parts:
            raise DocumentParseError(
                f"Path contains '..' traversal: {p}",
                details={"path": str(p)},
            )
        try:
            resolved = p.resolve()
        except (OSError, ValueError) as exc:
            raise DocumentParseError(
                f"Could not resolve path: {p}",
                details={"path": str(p)},
                cause=exc,
            ) from exc
        allowed_roots = [Path.cwd().resolve(), Path(tempfile.gettempdir()).resolve()]
        for root in allowed_roots:
            try:
                resolved.relative_to(root)
                return  # safe
            except ValueError:
                continue
        raise DocumentParseError(
            f"Path is outside allowed roots: {p}",
            details={"path": str(p), "allowed_roots": [str(r) for r in allowed_roots]},
        )

    def _parse(self, raw: bytes, content_type: str) -> str:
        ct = (content_type or "").lower().split(";")[0].strip()
        try:
            if ct in ("text/plain", "text/markdown"):
                return raw.decode("utf-8", errors="strict")
            if ct == "application/json":
                obj = json.loads(raw.decode("utf-8"))
                return _json_to_text(obj)
            if ct in ("text/html", "application/xhtml+xml"):
                return _strip_html(raw.decode("utf-8", errors="replace"))
            if ct == "application/pdf":
                return _pdf_to_text(raw)
            # Fallback: try utf-8 then latin-1.
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1")
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError(
                f"Failed to parse content of type {ct}: {exc}",
                details={"content_type": ct},
                cause=exc,
            ) from exc


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".json": "application/json",
        ".html": "text/html",
        ".htm": "text/html",
        ".pdf": "application/pdf",
    }.get(ext, "application/octet-stream")


def _json_to_text(obj: Any, depth: int = 0) -> str:
    """Flatten JSON into a textual representation suitable for chunking."""
    if depth > 8:
        return str(obj)
    if isinstance(obj, dict):
        parts: list[str] = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                parts.append(f"{k}: {_json_to_text(v, depth + 1)}")
            else:
                parts.append(f"{k}: {v}")
        return "\n".join(parts)
    if isinstance(obj, list):
        return "\n".join(_json_to_text(v, depth + 1) for v in obj)
    return str(obj)


def _strip_html(html: str) -> str:
    no_tags = _HTML_TAG_RE.sub(" ", html)
    return _HTML_WS_RE.sub(" ", no_tags).strip()


def _pdf_to_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise DocumentParseError(
            "PDF parsing requires `pypdf`; install it or supply text/markdown.",
        ) from exc
    import io

    reader = PdfReader(io.BytesIO(raw))
    chunks: list[str] = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(chunks).strip()


__all__ = ["DocumentLoader"]


# Keep urlparse import referenced (used for future URL loader)
_ = urlparse
