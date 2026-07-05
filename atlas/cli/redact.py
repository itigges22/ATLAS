"""Private-value filtering: masks credential-shaped values before they
reach a serialized sink (logs, error details, diagnostics).

CANONICAL COPY NOTICE: this file exists byte-identically in three
places — geometric-lens/geometric_lens/, sandbox/, and v3-service/ —
because each service ships as a separate container without a shared
package. tests/contracts/test_private_value_filtering.py enforces that
the copies stay identical and that each passes the shared fixture
corpus (tests/fixtures/private_value_fixtures.json), which the Go
implementation (proxy/private_values.go) also passes. Edit all copies
together.

Patterns are deliberately conservative (assignment/header/key-block
shapes with secret-ish key names) so ordinary content — "timeout=30",
token counts, health URLs — passes through untouched.
"""

import logging
import re

PLACEHOLDER = "[FILTERED]"

_ASSIGNMENT = re.compile(
    r'(?i)([A-Z0-9_.-]*(?:api[_-]?key|apikey|token|secret|password'
    r'|passwd|credential|access[_-]?key)[A-Z0-9_.-]*"?\s*[=:]\s*"?)'
    r'([^\s"\',;&]+)')
_BEARER = re.compile(r'(?i)(bearer\s+)([A-Za-z0-9._~+/=-]+)')
_URL_PASSWORD = re.compile(r'(://[^/:@\s]+:)([^@\s]+)(@)')
_PRIVATE_KEY_BLOCK = re.compile(
    r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----',
    re.S)


def filter_private_values(text: str) -> str:
    """Mask credential-shaped substrings in text."""
    if not text:
        return text
    text = _PRIVATE_KEY_BLOCK.sub(PLACEHOLDER, text)
    text = _ASSIGNMENT.sub(r'\g<1>' + PLACEHOLDER, text)
    text = _BEARER.sub(r'\g<1>' + PLACEHOLDER, text)
    text = _URL_PASSWORD.sub(r'\g<1>' + PLACEHOLDER + r'\g<3>', text)
    return text


class PrivateValueLogFilter(logging.Filter):
    """Attach to a logger (or root) so every record is filtered before
    any handler serializes it: logger.addFilter(PrivateValueLogFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            filtered = filter_private_values(msg)
            if filtered != msg:
                record.msg = filtered
                record.args = ()
        except Exception:
            pass  # a filtering failure must never suppress the log line
        return True
