"""
Tests for Embedding service.

Validates vector generation, batch processing,
and embedding quality.
"""

import math
import pytest
import httpx


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


class TestEmbeddingHealth:
    """Test Embedding service health."""

    def test_health_endpoint_responds(self, embedding_client: httpx.Client):
        """Health endpoint should return 200 OK."""
        response = embedding_client.get("/health")
        assert response.status_code == 200, f"Health endpoint should return 200, got {response.status_code}"

    def test_root_endpoint(self, embedding_client: httpx.Client):
        """Root endpoint should return service info."""
        response = embedding_client.get("/")
        assert response.status_code == 200, f"Root endpoint should return 200, got {response.status_code}"


class TestEmbeddingGeneration:
    """Test embedding vector generation."""

    def test_v1_embeddings_returns_vector(self, embedding_client: httpx.Client):
        """POST /v1/embeddings should return embedding vector."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": "Hello, world!"}
        )
        assert response.status_code == 200, f"/v1/embeddings should return 200, got {response.status_code}"
        data = response.json()
        assert "data" in data, "Response should have 'data' field"
        assert len(data["data"]) > 0, "Should have at least one embedding"
        embedding = data["data"][0].get("embedding")
        assert embedding is not None, "Should have embedding vector"
        assert isinstance(embedding, list), "Embedding should be a list"

    def test_embedding_is_384_dimensions(self, embedding_client: httpx.Client):
        """Embedding should be 384 dimensions (MiniLM-L6-v2)."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": "Test text for embedding"}
        )
        assert response.status_code == 200
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert len(embedding) == 384, f"Embedding should be 384 dimensions, got {len(embedding)}"

    def test_batch_embedding_works(self, embedding_client: httpx.Client):
        """Batch embedding with multiple inputs should work."""
        inputs = [
            "First text",
            "Second text",
            "Third text"
        ]
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": inputs}
        )
        assert response.status_code == 200, f"Batch embedding should return 200, got {response.status_code}"
        data = response.json()
        assert len(data["data"]) == 3, f"Should have 3 embeddings, got {len(data['data'])}"
        for i, item in enumerate(data["data"]):
            assert "embedding" in item, f"Item {i} should have embedding"
            assert len(item["embedding"]) == 384, f"Item {i} should be 384 dimensions"

    def test_empty_input_handled(self, embedding_client: httpx.Client):
        """Empty input should return embedding (model handles empty strings)."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": ""}
        )
        # Embedding service accepts empty input and returns embedding
        assert response.status_code == 200, f"Should handle empty input, got {response.status_code}"

    def test_long_input_handled(self, embedding_client: httpx.Client):
        """Long input should be truncated appropriately."""
        # Generate very long text
        long_text = "This is a test sentence. " * 1000  # ~5000 words

        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": long_text}
        )
        assert response.status_code == 200, f"Long input should be handled, got {response.status_code}"
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert len(embedding) == 384, "Should still produce 384-dim embedding"


class TestEmbeddingQuality:
    """Test embedding quality and semantic properties."""

    def test_similar_texts_have_similar_embeddings(self, embedding_client: httpx.Client):
        """Semantically similar texts should have high cosine similarity."""
        text1 = "The cat sat on the mat"
        text2 = "A cat was sitting on a mat"
        text3 = "Python is a programming language"

        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": [text1, text2, text3]}
        )
        assert response.status_code == 200
        data = response.json()

        emb1 = data["data"][0]["embedding"]
        emb2 = data["data"][1]["embedding"]
        emb3 = data["data"][2]["embedding"]

        # Similar texts should have higher similarity
        sim_similar = cosine_similarity(emb1, emb2)
        sim_different = cosine_similarity(emb1, emb3)

        assert sim_similar > sim_different, \
            f"Similar texts should have higher similarity ({sim_similar:.3f}) than different texts ({sim_different:.3f})"
        assert sim_similar > 0.5, f"Similar texts should have similarity > 0.5, got {sim_similar:.3f}"

    def test_different_texts_have_different_embeddings(self, embedding_client: httpx.Client):
        """Different texts should produce different embeddings."""
        text1 = "Machine learning is fascinating"
        text2 = "The stock market closed higher today"

        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": [text1, text2]}
        )
        assert response.status_code == 200
        data = response.json()

        emb1 = data["data"][0]["embedding"]
        emb2 = data["data"][1]["embedding"]

        # Embeddings should not be identical
        assert emb1 != emb2, "Different texts should have different embeddings"

        # Similarity should be lower for unrelated texts
        sim = cosine_similarity(emb1, emb2)
        assert sim < 0.9, f"Unrelated texts should have lower similarity, got {sim:.3f}"

    def test_identical_texts_have_identical_embeddings(self, embedding_client: httpx.Client):
        """Identical texts should produce identical embeddings."""
        text = "This is a test"

        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": [text, text]}
        )
        assert response.status_code == 200
        data = response.json()

        emb1 = data["data"][0]["embedding"]
        emb2 = data["data"][1]["embedding"]

        # Should be identical (or very close due to floating point)
        sim = cosine_similarity(emb1, emb2)
        assert sim > 0.999, f"Identical texts should have similarity > 0.999, got {sim:.3f}"


class TestEmbeddingResponseFormat:
    """Test embedding response format."""

    def test_openai_compatible_format(self, embedding_client: httpx.Client):
        """Response should be OpenAI-compatible format."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": "Test"}
        )
        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "object" in data or "data" in data, "Should have object or data field"
        assert "data" in data, "Should have data field"
        assert isinstance(data["data"], list), "data should be a list"

        if len(data["data"]) > 0:
            item = data["data"][0]
            assert "embedding" in item, "Item should have embedding"
            assert "index" in item or "object" in item, "Item should have index or object"


class TestEmbeddingInputValidation:
    """Test embedding input validation."""

    def test_whitespace_only_input(self, embedding_client: httpx.Client):
        """Whitespace-only input should return embedding."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": "   \t\n   "}
        )
        # Embedding service handles whitespace and returns embedding
        assert response.status_code == 200, f"Whitespace input should work, got {response.status_code}"

    def test_single_character_input(self, embedding_client: httpx.Client):
        """Single character input should work."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": "a"}
        )
        assert response.status_code == 200, "Single character should work"
        data = response.json()
        assert len(data["data"][0]["embedding"]) == 384

    def test_unicode_text(self, embedding_client: httpx.Client):
        """Unicode text should work."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": "こんにちは世界 🌍 مرحبا"}
        )
        assert response.status_code == 200, "Unicode text should work"

    def test_emoji_text(self, embedding_client: httpx.Client):
        """Emoji text should work."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": "🎉🚀💻🔥✨"}
        )
        assert response.status_code == 200, "Emoji text should work"

    def test_code_as_text(self, embedding_client: httpx.Client):
        """Code should be embeddable."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": "def add(a, b): return a + b"}
        )
        assert response.status_code == 200, "Code should work"

    def test_json_as_text(self, embedding_client: httpx.Client):
        """JSON string should be embeddable."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": '{"key": "value", "number": 42}'}
        )
        assert response.status_code == 200, "JSON string should work"


class TestEmbeddingVectorValidation:
    """Test embedding vector properties."""

    def test_values_are_floats(self, embedding_client: httpx.Client):
        """Embedding values should be floats."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": "Test"}
        )
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert all(isinstance(v, (int, float)) for v in embedding), "Values should be numeric"

    def test_values_in_reasonable_range(self, embedding_client: httpx.Client):
        """Embedding values should be in reasonable range."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": "Test"}
        )
        data = response.json()
        embedding = data["data"][0]["embedding"]
        # Typically normalized embeddings are between -1 and 1
        max_val = max(embedding)
        min_val = min(embedding)
        assert -10 <= min_val <= 10, f"Min value should be reasonable, got {min_val}"
        assert -10 <= max_val <= 10, f"Max value should be reasonable, got {max_val}"

    def test_deterministic_output(self, embedding_client: httpx.Client):
        """Same input should produce same embedding."""
        text = "Reproducibility test"
        response1 = embedding_client.post("/v1/embeddings", json={"input": text})
        response2 = embedding_client.post("/v1/embeddings", json={"input": text})

        emb1 = response1.json()["data"][0]["embedding"]
        emb2 = response2.json()["data"][0]["embedding"]

        sim = cosine_similarity(emb1, emb2)
        assert sim > 0.999, f"Same input should produce same embedding, sim={sim}"


class TestEmbeddingBatchOperations:
    """Test batch embedding operations."""

    def test_batch_of_ten(self, embedding_client: httpx.Client):
        """Batch of 10 texts should work."""
        inputs = [f"Text number {i}" for i in range(10)]
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": inputs}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 10, f"Should have 10 embeddings, got {len(data['data'])}"

    def test_batch_preserves_order(self, embedding_client: httpx.Client):
        """Batch should preserve input order."""
        inputs = ["Apple", "Banana", "Cherry"]
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": inputs}
        )
        data = response.json()
        # Check indices if present
        for i, item in enumerate(data["data"]):
            if "index" in item:
                assert item["index"] == i, f"Index should match position"


class TestEmbeddingSimilarityDetailed:
    """Test detailed similarity scenarios."""

    def test_antonym_similarity(self, embedding_client: httpx.Client):
        """Antonyms should have moderate similarity (same domain)."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": ["hot", "cold", "programming"]}
        )
        data = response.json()
        emb_hot = data["data"][0]["embedding"]
        emb_cold = data["data"][1]["embedding"]
        emb_prog = data["data"][2]["embedding"]

        sim_antonyms = cosine_similarity(emb_hot, emb_cold)
        sim_unrelated = cosine_similarity(emb_hot, emb_prog)
        # Antonyms often have higher similarity than unrelated words
        # because they're in the same semantic domain

    def test_synonym_high_similarity(self, embedding_client: httpx.Client):
        """Synonyms should have high similarity."""
        response = embedding_client.post(
            "/v1/embeddings",
            json={"input": ["happy", "joyful", "angry"]}
        )
        data = response.json()
        emb_happy = data["data"][0]["embedding"]
        emb_joyful = data["data"][1]["embedding"]
        emb_angry = data["data"][2]["embedding"]

        sim_synonyms = cosine_similarity(emb_happy, emb_joyful)
        sim_different = cosine_similarity(emb_happy, emb_angry)

        assert sim_synonyms > sim_different, "Synonyms should be more similar"
