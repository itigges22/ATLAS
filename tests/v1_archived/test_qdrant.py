"""
Tests for Qdrant vector database.

Validates collection operations, vector storage,
and similarity search.
"""

import uuid
import pytest
import httpx


class TestQdrantHealth:
    """Test Qdrant service health."""

    def test_health_endpoint_responds(self, qdrant_client: httpx.Client):
        """Qdrant should respond to requests."""
        # Qdrant doesn't have a /health endpoint, but / or /collections works
        response = qdrant_client.get("/collections")
        assert response.status_code == 200, f"Qdrant should respond, got {response.status_code}"


class TestQdrantCollections:
    """Test collection operations."""

    def test_can_list_collections(self, qdrant_client: httpx.Client):
        """Should be able to list collections."""
        response = qdrant_client.get("/collections")
        assert response.status_code == 200, f"List collections should return 200, got {response.status_code}"
        data = response.json()
        assert "result" in data, "Response should have 'result' field"
        assert "collections" in data["result"], "Result should have 'collections' field"

    def test_can_create_collection(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should be able to create a test collection."""
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        response = qdrant_client.put(
            f"/collections/{collection_name}",
            json={
                "vectors": {
                    "size": 384,
                    "distance": "Cosine"
                }
            }
        )
        assert response.status_code in [200, 201], f"Create collection should succeed, got {response.status_code}"

        # Verify collection exists
        list_response = qdrant_client.get("/collections")
        collections = [c["name"] for c in list_response.json()["result"]["collections"]]
        assert collection_name in collections, f"Collection {collection_name} should exist"

    def test_can_delete_collection(
        self,
        qdrant_client: httpx.Client
    ):
        """Should be able to delete a collection."""
        collection_name = f"test_delete_{uuid.uuid4().hex[:8]}"

        # Create collection first
        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        # Delete collection
        response = qdrant_client.delete(f"/collections/{collection_name}")
        assert response.status_code == 200, f"Delete collection should return 200, got {response.status_code}"

        # Verify collection is gone
        list_response = qdrant_client.get("/collections")
        collections = [c["name"] for c in list_response.json()["result"]["collections"]]
        assert collection_name not in collections, f"Collection {collection_name} should be deleted"


class TestQdrantVectorOperations:
    """Test vector storage and retrieval."""

    def test_can_insert_vectors(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should be able to insert vectors with payload."""
        collection_name = f"test_insert_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        # Create collection
        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        # Insert vectors
        points = [
            {
                "id": 1,
                "vector": [0.1] * 384,
                "payload": {"text": "First document", "source": "test"}
            },
            {
                "id": 2,
                "vector": [0.2] * 384,
                "payload": {"text": "Second document", "source": "test"}
            }
        ]

        response = qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": points},
            params={"wait": "true"}
        )
        assert response.status_code == 200, f"Insert should succeed, got {response.status_code}"

        # Verify count
        info_response = qdrant_client.get(f"/collections/{collection_name}")
        assert info_response.status_code == 200
        count = info_response.json()["result"]["points_count"]
        assert count == 2, f"Should have 2 points, got {count}"

    def test_can_search_vectors(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should be able to search vectors by similarity."""
        collection_name = f"test_search_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        # Create collection and insert vectors
        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        # Create distinct vectors
        vec1 = [1.0] + [0.0] * 383  # Points in one direction
        vec2 = [0.0] + [1.0] + [0.0] * 382  # Points in another direction
        vec3 = [0.9] + [0.1] + [0.0] * 382  # Similar to vec1

        points = [
            {"id": 1, "vector": vec1, "payload": {"label": "first"}},
            {"id": 2, "vector": vec2, "payload": {"label": "second"}},
            {"id": 3, "vector": vec3, "payload": {"label": "similar_to_first"}}
        ]

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": points},
            params={"wait": "true"}
        )

        # Search for vectors similar to vec1
        search_response = qdrant_client.post(
            f"/collections/{collection_name}/points/search",
            json={
                "vector": vec1,
                "limit": 3,
                "with_payload": True
            }
        )
        assert search_response.status_code == 200

        results = search_response.json()["result"]
        assert len(results) > 0, "Should find results"

        # First result should be the most similar (id 1 or 3)
        first_id = results[0]["id"]
        assert first_id in [1, 3], f"Most similar should be id 1 or 3, got {first_id}"

    def test_can_filter_search_by_payload(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should be able to filter search by payload fields."""
        collection_name = f"test_filter_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        # Create collection
        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        # Insert vectors with different categories
        points = [
            {"id": 1, "vector": [0.1] * 384, "payload": {"category": "A", "text": "doc1"}},
            {"id": 2, "vector": [0.11] * 384, "payload": {"category": "B", "text": "doc2"}},
            {"id": 3, "vector": [0.12] * 384, "payload": {"category": "A", "text": "doc3"}}
        ]

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": points},
            params={"wait": "true"}
        )

        # Search with filter
        search_response = qdrant_client.post(
            f"/collections/{collection_name}/points/search",
            json={
                "vector": [0.1] * 384,
                "limit": 10,
                "filter": {
                    "must": [
                        {"key": "category", "match": {"value": "A"}}
                    ]
                },
                "with_payload": True
            }
        )
        assert search_response.status_code == 200

        results = search_response.json()["result"]
        # Should only return category A documents
        for result in results:
            assert result["payload"]["category"] == "A", f"Filtered results should be category A"

    def test_collection_persists_across_requests(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Collection and data should persist across requests."""
        collection_name = f"test_persist_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        # Create and populate collection
        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={
                "points": [
                    {"id": 1, "vector": [0.5] * 384, "payload": {"test": "data"}}
                ]
            },
            params={"wait": "true"}
        )

        # Make separate request to verify persistence
        info_response = qdrant_client.get(f"/collections/{collection_name}")
        assert info_response.status_code == 200
        count = info_response.json()["result"]["points_count"]
        assert count == 1, "Data should persist"

        # Retrieve the point
        get_response = qdrant_client.get(
            f"/collections/{collection_name}/points/1",
            params={"with_payload": "true", "with_vector": "true"}
        )
        assert get_response.status_code == 200
        point = get_response.json()["result"]
        assert point["payload"]["test"] == "data", "Payload should persist"


class TestQdrantCollectionConfig:
    """Test collection configuration options."""

    def test_euclidean_distance(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should support Euclidean distance metric."""
        collection_name = f"test_euclidean_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        response = qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 128, "distance": "Euclid"}}
        )
        assert response.status_code in [200, 201]

    def test_dot_product_distance(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should support Dot product distance metric."""
        collection_name = f"test_dot_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        response = qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 128, "distance": "Dot"}}
        )
        assert response.status_code in [200, 201]

    def test_collection_info_returns_config(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Collection info should return configuration."""
        collection_name = f"test_info_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 256, "distance": "Cosine"}}
        )

        response = qdrant_client.get(f"/collections/{collection_name}")
        assert response.status_code == 200
        data = response.json()["result"]
        assert "config" in data or "vectors_count" in data or "points_count" in data

    def test_custom_vector_size(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should support custom vector sizes."""
        collection_name = f"test_size_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        response = qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 1536, "distance": "Cosine"}}  # OpenAI size
        )
        assert response.status_code in [200, 201]


class TestQdrantPointOperations:
    """Test individual point operations."""

    def test_get_point_by_id(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should retrieve point by ID."""
        collection_name = f"test_getpoint_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": [{"id": 42, "vector": [0.1] * 384, "payload": {"name": "test"}}]},
            params={"wait": "true"}
        )

        response = qdrant_client.get(f"/collections/{collection_name}/points/42")
        assert response.status_code == 200
        data = response.json()["result"]
        assert data["id"] == 42

    def test_update_point_payload(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should update point payload."""
        collection_name = f"test_update_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": [{"id": 1, "vector": [0.1] * 384, "payload": {"status": "old"}}]},
            params={"wait": "true"}
        )

        # Update payload
        response = qdrant_client.post(
            f"/collections/{collection_name}/points/payload",
            json={
                "payload": {"status": "updated"},
                "points": [1]
            },
            params={"wait": "true"}
        )
        assert response.status_code == 200

    def test_delete_points(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should delete points by ID."""
        collection_name = f"test_delpoint_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": [
                {"id": 1, "vector": [0.1] * 384},
                {"id": 2, "vector": [0.2] * 384}
            ]},
            params={"wait": "true"}
        )

        # Delete point
        response = qdrant_client.post(
            f"/collections/{collection_name}/points/delete",
            json={"points": [1]},
            params={"wait": "true"}
        )
        assert response.status_code == 200

        # Verify count
        info = qdrant_client.get(f"/collections/{collection_name}").json()["result"]
        assert info["points_count"] == 1

    def test_upsert_points(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should upsert (insert or update) points."""
        collection_name = f"test_upsert_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        # Initial insert
        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": [{"id": 1, "vector": [0.1] * 384, "payload": {"v": 1}}]},
            params={"wait": "true"}
        )

        # Upsert same ID with new payload
        response = qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": [{"id": 1, "vector": [0.2] * 384, "payload": {"v": 2}}]},
            params={"wait": "true"}
        )
        assert response.status_code == 200

        # Verify only one point
        info = qdrant_client.get(f"/collections/{collection_name}").json()["result"]
        assert info["points_count"] == 1


class TestQdrantSearchAdvanced:
    """Test advanced search capabilities."""

    def test_search_with_score_threshold(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should support score threshold filtering."""
        collection_name = f"test_threshold_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": [
                {"id": 1, "vector": [1.0] + [0.0] * 383},
                {"id": 2, "vector": [0.5] + [0.5] + [0.0] * 382}
            ]},
            params={"wait": "true"}
        )

        response = qdrant_client.post(
            f"/collections/{collection_name}/points/search",
            json={
                "vector": [1.0] + [0.0] * 383,
                "limit": 10,
                "score_threshold": 0.9
            }
        )
        assert response.status_code == 200

    def test_search_with_offset(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should support offset in search."""
        collection_name = f"test_offset_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        points = [{"id": i, "vector": [0.1 * i] + [0.0] * 383} for i in range(1, 6)]
        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": points},
            params={"wait": "true"}
        )

        response = qdrant_client.post(
            f"/collections/{collection_name}/points/search",
            json={
                "vector": [0.5] + [0.0] * 383,
                "limit": 2,
                "offset": 1
            }
        )
        assert response.status_code == 200
        results = response.json()["result"]
        assert len(results) <= 2

    def test_filter_by_range(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should support range filters."""
        collection_name = f"test_range_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        points = [
            {"id": 1, "vector": [0.1] * 384, "payload": {"score": 10}},
            {"id": 2, "vector": [0.1] * 384, "payload": {"score": 50}},
            {"id": 3, "vector": [0.1] * 384, "payload": {"score": 90}}
        ]
        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": points},
            params={"wait": "true"}
        )

        response = qdrant_client.post(
            f"/collections/{collection_name}/points/search",
            json={
                "vector": [0.1] * 384,
                "limit": 10,
                "filter": {
                    "must": [
                        {"key": "score", "range": {"gte": 20, "lte": 80}}
                    ]
                },
                "with_payload": True
            }
        )
        assert response.status_code == 200
        results = response.json()["result"]
        for r in results:
            assert 20 <= r["payload"]["score"] <= 80

    def test_filter_must_not(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should support must_not filters."""
        collection_name = f"test_mustnot_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        points = [
            {"id": 1, "vector": [0.1] * 384, "payload": {"type": "spam"}},
            {"id": 2, "vector": [0.1] * 384, "payload": {"type": "valid"}},
            {"id": 3, "vector": [0.1] * 384, "payload": {"type": "valid"}}
        ]
        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": points},
            params={"wait": "true"}
        )

        response = qdrant_client.post(
            f"/collections/{collection_name}/points/search",
            json={
                "vector": [0.1] * 384,
                "limit": 10,
                "filter": {
                    "must_not": [
                        {"key": "type", "match": {"value": "spam"}}
                    ]
                },
                "with_payload": True
            }
        )
        assert response.status_code == 200
        results = response.json()["result"]
        for r in results:
            assert r["payload"]["type"] != "spam"


class TestQdrantPayloadTypes:
    """Test different payload types."""

    def test_string_payload(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should handle string payloads."""
        collection_name = f"test_string_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": [{"id": 1, "vector": [0.1] * 384, "payload": {"text": "hello world"}}]},
            params={"wait": "true"}
        )

        response = qdrant_client.get(f"/collections/{collection_name}/points/1")
        assert response.json()["result"]["payload"]["text"] == "hello world"

    def test_numeric_payload(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should handle numeric payloads."""
        collection_name = f"test_numeric_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": [{"id": 1, "vector": [0.1] * 384, "payload": {"count": 42, "score": 3.14}}]},
            params={"wait": "true"}
        )

        response = qdrant_client.get(f"/collections/{collection_name}/points/1")
        payload = response.json()["result"]["payload"]
        assert payload["count"] == 42
        assert abs(payload["score"] - 3.14) < 0.01

    def test_array_payload(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should handle array payloads."""
        collection_name = f"test_array_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": [{"id": 1, "vector": [0.1] * 384, "payload": {"tags": ["a", "b", "c"]}}]},
            params={"wait": "true"}
        )

        response = qdrant_client.get(f"/collections/{collection_name}/points/1")
        assert response.json()["result"]["payload"]["tags"] == ["a", "b", "c"]

    def test_nested_payload(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should handle nested payloads."""
        collection_name = f"test_nested_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": [{"id": 1, "vector": [0.1] * 384, "payload": {"meta": {"author": "test", "version": 1}}}]},
            params={"wait": "true"}
        )

        response = qdrant_client.get(f"/collections/{collection_name}/points/1")
        meta = response.json()["result"]["payload"]["meta"]
        assert meta["author"] == "test"
        assert meta["version"] == 1


class TestQdrantBulkOperations:
    """Test bulk operations."""

    def test_bulk_insert_100_points(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should handle bulk insert of 100 points."""
        collection_name = f"test_bulk100_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        points = [{"id": i, "vector": [0.01 * i] * 384, "payload": {"idx": i}} for i in range(100)]
        response = qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": points},
            params={"wait": "true"}
        )
        assert response.status_code == 200

        info = qdrant_client.get(f"/collections/{collection_name}").json()["result"]
        assert info["points_count"] == 100

    def test_scroll_points(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should support scrolling through points."""
        collection_name = f"test_scroll_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        points = [{"id": i, "vector": [0.1] * 384} for i in range(20)]
        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": points},
            params={"wait": "true"}
        )

        response = qdrant_client.post(
            f"/collections/{collection_name}/points/scroll",
            json={"limit": 10, "with_payload": True}
        )
        assert response.status_code == 200
        data = response.json()["result"]
        assert len(data["points"]) == 10

    def test_count_points(
        self,
        qdrant_client: httpx.Client,
        cleanup_qdrant_collections: list
    ):
        """Should count points with filter."""
        collection_name = f"test_count_{uuid.uuid4().hex[:8]}"
        cleanup_qdrant_collections.append(collection_name)

        qdrant_client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": 384, "distance": "Cosine"}}
        )

        points = [
            {"id": 1, "vector": [0.1] * 384, "payload": {"active": True}},
            {"id": 2, "vector": [0.1] * 384, "payload": {"active": False}},
            {"id": 3, "vector": [0.1] * 384, "payload": {"active": True}}
        ]
        qdrant_client.put(
            f"/collections/{collection_name}/points",
            json={"points": points},
            params={"wait": "true"}
        )

        response = qdrant_client.post(
            f"/collections/{collection_name}/points/count",
            json={
                "filter": {"must": [{"key": "active", "match": {"value": True}}]},
                "exact": True
            }
        )
        assert response.status_code == 200
        count = response.json()["result"]["count"]
        assert count == 2
