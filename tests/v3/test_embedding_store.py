"""Tests for V3 binary embedding store."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.v3.embedding_store import EmbeddingWriter, EmbeddingReader


class TestEmbeddingStore:
    def test_write_and_read_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            path = Path(f.name)

        emb = [float(i) for i in range(64)]  # small dim for testing
        with EmbeddingWriter(path) as writer:
            writer.write("task_001", 0, "PASS", emb)
            writer.write("task_001", 1, "FAIL", emb)
            writer.write("task_002", 0, "UNKNOWN", emb)

        reader = EmbeddingReader(path)
        records = reader.read_all()

        assert len(records) == 3
        assert records[0]["task_id"] == "task_001"
        assert records[0]["candidate_index"] == 0
        assert records[0]["label"] == "PASS"
        assert len(records[0]["embedding"]) == 64
        assert records[0]["embedding"][0] == 0.0
        assert records[0]["embedding"][63] == 63.0

        assert records[1]["label"] == "FAIL"
        assert records[2]["label"] == "UNKNOWN"
        assert records[2]["task_id"] == "task_002"

        path.unlink()

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            path = Path(f.name)

        reader = EmbeddingReader(path)
        records = reader.read_all()
        assert len(records) == 0

        path.unlink()

    def test_large_embedding(self):
        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            path = Path(f.name)

        emb = [float(i) * 0.001 for i in range(5120)]  # real dim
        with EmbeddingWriter(path) as writer:
            writer.write("LCB_3033", 0, "PASS", emb)

        reader = EmbeddingReader(path)
        records = reader.read_all()
        assert len(records) == 1
        assert len(records[0]["embedding"]) == 5120
        assert abs(records[0]["embedding"][5119] - 5.119) < 0.001

        path.unlink()

    def test_unicode_task_id(self):
        with tempfile.NamedTemporaryFile(suffix=".emb", delete=False) as f:
            path = Path(f.name)

        emb = [1.0, 2.0, 3.0]
        with EmbeddingWriter(path) as writer:
            writer.write("LCB_abc301_a", 0, "PASS", emb)

        reader = EmbeddingReader(path)
        records = reader.read_all()
        assert records[0]["task_id"] == "LCB_abc301_a"

        path.unlink()
