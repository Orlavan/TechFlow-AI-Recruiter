"""
Embeddings Module with ChromaDB Vector Store
Handles document embedding and retrieval for RAG-based information retrieval.
Supports both PDF and TXT job descriptions.
"""

import os
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configuration
CHROMA_PATH = "chroma_db"
JOB_DESCRIPTION_TXT = "job_description.txt"
JOB_DESCRIPTION_PDF = "Python Developer Job Description.pdf"


class EmbeddingsManager:
    """
    Manages document embeddings and vector store operations.
    Uses ChromaDB for persistent vector storage.
    """

    def __init__(self, persist_directory: str = CHROMA_PATH):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self._vectorstore = None

    @property
    def vectorstore(self) -> Chroma:
        """Lazy-loads the vector store."""
        if self._vectorstore is None:
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                self._vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                self._vectorstore = self.create_vectorstore()
        return self._vectorstore

    def create_vectorstore(self, documents: Optional[List[Document]] = None) -> Chroma:
        """
        Creates and seeds the vector store with job description documents.
        """
        if documents is None:
            documents = self._load_job_description()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = text_splitter.split_documents(documents)

        # Create ChromaDB vector store
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        print(f"Vector store created with {len(docs)} document chunks.")
        return vectorstore

    def _load_job_description(self) -> List[Document]:
        """
        Loads job description from PDF or TXT file.
        Prioritizes PDF if available.
        """
        documents = []

        # Try PDF first
        if os.path.exists(JOB_DESCRIPTION_PDF):
            try:
                loader = PyPDFLoader(JOB_DESCRIPTION_PDF)
                documents = loader.load()
                print(f"Loaded job description from PDF: {JOB_DESCRIPTION_PDF}")
                return documents
            except Exception as e:
                print(f"Failed to load PDF: {e}. Falling back to TXT.")

        # Fall back to TXT
        if os.path.exists(JOB_DESCRIPTION_TXT):
            loader = TextLoader(JOB_DESCRIPTION_TXT, encoding='utf-8')
            documents = loader.load()
            print(f"Loaded job description from TXT: {JOB_DESCRIPTION_TXT}")
            return documents

        # Create default document if no file found
        print("Warning: No job description file found. Using default content.")
        default_content = """
        Senior Python Developer Position at TechFlow Solutions

        Location: Tel Aviv (Hybrid)
        Department: Engineering

        Requirements:
        - 3+ years of professional Python development experience
        - Experience with web frameworks (Django, Flask, FastAPI)
        - Cloud services experience (AWS, Google Cloud, Azure)
        - Docker and containerization knowledge
        - SQL and database design proficiency

        Responsibilities:
        - Design and maintain Python backend services
        - Develop APIs and integrate with databases
        - Collaborate with frontend developers
        - Ensure code quality and performance

        Benefits:
        - Competitive compensation
        - Professional growth opportunities
        - Hybrid work model
        """
        return [Document(page_content=default_content, metadata={"source": "default"})]

    def query(self, query: str, k: int = 3) -> List[Document]:
        """
        Queries the vector store for relevant documents.

        Args:
            query: Search query
            k: Number of documents to return

        Returns:
            List of relevant documents
        """
        return self.vectorstore.similarity_search(query, k=k)

    def query_with_scores(self, query: str, k: int = 3) -> List[tuple]:
        """
        Queries with relevance scores.
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def get_relevant_context(self, query: str, max_chars: int = 1500) -> str:
        """
        Gets relevant context for a query as a single string.
        Useful for prompt injection.
        """
        docs = self.query(query, k=3)
        context_parts = []
        total_chars = 0

        for doc in docs:
            content = doc.page_content.strip()
            if total_chars + len(content) <= max_chars:
                context_parts.append(content)
                total_chars += len(content)
            else:
                remaining = max_chars - total_chars
                if remaining > 100:
                    context_parts.append(content[:remaining] + "...")
                break

        return "\n\n".join(context_parts)

    def add_documents(self, documents: List[Document]):
        """Adds additional documents to the vector store."""
        self.vectorstore.add_documents(documents)

    def clear_vectorstore(self):
        """Clears the vector store (for re-initialization)."""
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self._vectorstore = None
        print("Vector store cleared.")


# Convenience functions for backwards compatibility
def get_vectorstore() -> Chroma:
    """Returns the vector store instance."""
    manager = EmbeddingsManager()
    return manager.vectorstore


def create_and_seed_vectorstore() -> Chroma:
    """Creates and seeds a new vector store."""
    manager = EmbeddingsManager()
    manager.clear_vectorstore()
    return manager.create_vectorstore()


def query_info(query: str) -> str:
    """
    Queries the vector store and returns relevant information.
    """
    manager = EmbeddingsManager()
    docs = manager.query(query, k=2)

    if docs:
        return "\n\n".join([doc.page_content for doc in docs])
    return "I don't have specific information about that in the job description."


def init_embeddings():
    """
    Initializes the embeddings database.
    Run this once before starting the application.
    """
    manager = EmbeddingsManager()
    manager.clear_vectorstore()
    manager.create_vectorstore()
    print("Embeddings initialized successfully.")


if __name__ == "__main__":
    init_embeddings()
