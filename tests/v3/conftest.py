"""Path setup for the V3 stage tests: `stages.*` lives in v3-service/
(flat sibling-import service layout), so v3-service must be on sys.path —
same insertion the tests/v3-service suite does per-file."""

import sys
from pathlib import Path

_V3_SERVICE = Path(__file__).resolve().parents[2] / "v3-service"
if str(_V3_SERVICE) not in sys.path:
    sys.path.insert(0, str(_V3_SERVICE))
