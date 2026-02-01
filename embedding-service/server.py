from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding Service", version="1.0.0")

# Load model at startup
logger.info("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
logger.info("Model loaded successfully")


class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "all-MiniLM-L6-v2"


class EmbedResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Embedding Service", "model": "all-MiniLM-L6-v2"}


@app.post("/embed")
async def embed(request: EmbedRequest):
    """Generate embeddings - simple endpoint"""
    try:
        inputs = request.input if isinstance(request.input, list) else [request.input]
        embeddings = model.encode(inputs, convert_to_numpy=True)

        return {
            "embeddings": embeddings.tolist()
        }
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def embeddings_openai(request: EmbedRequest):
    """OpenAI-compatible embeddings endpoint"""
    try:
        inputs = request.input if isinstance(request.input, list) else [request.input]
        embeddings = model.encode(inputs, convert_to_numpy=True)

        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": emb.tolist()
            }
            for i, emb in enumerate(embeddings)
        ]

        return EmbedResponse(
            object="list",
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": sum(len(text.split()) for text in inputs),
                "total_tokens": sum(len(text.split()) for text in inputs)
            }
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
