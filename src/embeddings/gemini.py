from openai import OpenAI
from neo4j_graphrag.embeddings.base import Embedder

from src.config import settings

GEMINI_INDEX_NAME = "chunk_embeddings"
GEMINI_EMBEDDING_DIMS = 3072


class GeminiEmbedder(Embedder):
    """Embedder for google/gemini-embedding-2 via OpenRouter.

    Forces encoding_format='float' — OpenRouter returns floats for Gemini
    but the openai SDK requests base64 by default, causing a silent parse failure.
    """

    def __init__(self):
        self._client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )

    def embed_query(self, text: str, **kwargs) -> list[float]:
        resp = self._client.embeddings.create(
            input=text,
            model=settings.embedding_model,
            encoding_format="float",
        )
        return resp.data[0].embedding
