import httpx
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config import config
from chunker import Chunk

logger = logging.getLogger(__name__)

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384


class VectorStore:
    """Interface to Qdrant vector database and embedding service."""

    def __init__(self):
        self.qdrant = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port
        )
        self.embedding_url = config.embedding.base_url

    def _collection_name(self, project_id: str) -> str:
        """Generate collection name for a project."""
        return f"project_{project_id}"

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from the embedding service."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.embedding_url}/embed",
                json={"input": texts}
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

    def create_collection(self, project_id: str) -> bool:
        """Create a new collection for a project."""
        collection_name = self._collection_name(project_id)

        # Delete if exists
        try:
            self.qdrant.delete_collection(collection_name)
        except Exception:
            pass

        # Create new collection
        self.qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection {collection_name}")
        return True

    async def index_chunks(self, project_id: str, chunks: List[Chunk]) -> int:
        """Index chunks into the vector store."""
        if not chunks:
            return 0

        collection_name = self._collection_name(project_id)

        # Create collection
        self.create_collection(project_id)

        # Get embeddings in batches
        batch_size = 32
        points = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]

            embeddings = await self.get_embeddings(texts)

            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                point = PointStruct(
                    id=i + j,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "content": chunk.content,
                        "language": chunk.language,
                        "chunk_type": chunk.chunk_type,
                        "symbol_name": chunk.symbol_name
                    }
                )
                points.append(point)

        # Upsert all points
        self.qdrant.upsert(
            collection_name=collection_name,
            points=points
        )

        logger.info(f"Indexed {len(points)} chunks for project {project_id}")
        return len(points)

    async def search(
        self,
        project_id: str,
        query: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for relevant chunks."""
        collection_name = self._collection_name(project_id)

        # Get query embedding
        embeddings = await self.get_embeddings([query])
        query_vector = embeddings[0]

        # Search
        results = self.qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        return [
            {
                "score": hit.score,
                **hit.payload
            }
            for hit in results
        ]

    def delete_collection(self, project_id: str) -> bool:
        """Delete a project's collection."""
        collection_name = self._collection_name(project_id)
        try:
            self.qdrant.delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def collection_exists(self, project_id: str) -> bool:
        """Check if a collection exists."""
        collection_name = self._collection_name(project_id)
        try:
            self.qdrant.get_collection(collection_name)
            return True
        except Exception:
            return False


# Global vector store instance
vector_store = VectorStore()
