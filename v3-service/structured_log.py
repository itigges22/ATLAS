"""Structured JSON logging + correlation IDs (Python services).

CANONICAL COPY NOTICE: byte-identical copies live in geometric-lens/
geometric_lens/, sandbox/, and v3-service/ (separate containers, no
shared package). tests/contracts/test_structured_log_contract.py
enforces they stay identical. Edit all copies together.

`install(service, root_logger)` attaches a JSON formatter when
ATLAS_LOG_FORMAT=json (else leaves the human format), plus the
private-value filter so records are masked before serialization. The
correlation ID for the current request is set via set_request_id() (from
an inbound X-ATLAS-Request-ID header) and included in every record.
"""

import json
import logging
import os
import threading

_LOCAL = threading.local()


def set_request_id(request_id):
    _LOCAL.request_id = request_id or ""


def get_request_id():
    return getattr(_LOCAL, "request_id", "")


class JsonFormatter(logging.Formatter):
    def __init__(self, service):
        super().__init__()
        self.service = service

    def format(self, record):
        rec = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = get_request_id()
        if rid:
            rec["request_id"] = rid
        if record.exc_info:
            rec["exc"] = self.formatException(record.exc_info)
        return json.dumps(rec)


def install(service, root_logger=None):
    """JSON format when ATLAS_LOG_FORMAT=json; always attach the
    private-value filter. Idempotent."""
    root = root_logger or logging.getLogger()
    # private-value masking (shared filter)
    try:
        from .private_values import PrivateValueLogFilter
    except ImportError:  # flat layout (sandbox/v3 copy)
        from private_values import PrivateValueLogFilter  # type: ignore
    if os.environ.get("ATLAS_LOG_FORMAT", "").lower() == "json":
        fmt = JsonFormatter(service)
        for h in root.handlers:
            h.setFormatter(fmt)
    for h in root.handlers:
        if not any(isinstance(f, PrivateValueLogFilter) for f in h.filters):
            h.addFilter(PrivateValueLogFilter())
