from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.embeddings import OpenAIEmbeddings

from src.config import settings


def get_embedder() -> Embedder:
    """Return the correct embedder based on EMBEDDING_MODEL in .env.

    Google models require encoding_format='float' via OpenRouter — the openai SDK
    defaults to base64, which causes a silent parse failure for Gemini responses.
    All other models use the standard OpenAIEmbeddings client.
    """
    if settings.embedding_model.startswith("google/"):
        from src.embeddings.gemini import GeminiEmbedder
        return GeminiEmbedder()
    return OpenAIEmbeddings(model=settings.embedding_model)
