"""Info Agent Module - RAG-based information retrieval and response generation."""
from app.modules.info_agent.info_agent import InfoAdvisor
from app.modules.info_agent.ingest import EmbeddingsManager, init_embeddings, query_info

__all__ = ["InfoAdvisor", "EmbeddingsManager", "init_embeddings", "query_info"]
