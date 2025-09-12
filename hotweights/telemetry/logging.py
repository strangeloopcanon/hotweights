from __future__ import annotations

"""Lightweight structured logging utilities.

Environment variables:
- HOTWEIGHTS_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default INFO)
"""

import logging
import os
from typing import Optional, Dict


_CONFIGURED = False


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.getenv("HOTWEIGHTS_LOG_LEVEL", "INFO").upper()
    try:
        lvl = getattr(logging, level)
    except Exception:
        lvl = logging.INFO
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str, context: Optional[Dict[str, object]] = None) -> logging.Logger:
    _configure_once()
    logger = logging.getLogger(name)
    if context:
        # Prepend context to messages via adapter
        class _Adapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):  # type: ignore[override]
                ctx = " ".join(f"{k}={v}" for k, v in context.items())
                return f"[{ctx}] {msg}", kwargs

        return _Adapter(logger, {})  # type: ignore[return-value]
    return logger

