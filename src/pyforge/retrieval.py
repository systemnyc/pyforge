"""Retrieve Python documentation passages from a persistent Chroma index.

        The embedding model and retriever are initialized when this module is
        imported. The embedding model is also reused for explanation comparison.
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

project_root = Path(__file__).resolve().parents[2]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Connect to the persisted python_docs collection created by ingest_docs.py.
# The embedding function converts search queries into vectors for comparison.
vector_store = Chroma(
    collection_name="python_docs",
    # Use the same embedding model as ingestion so queries match stored vectors.
    embedding_function=embedding_model,
    persist_directory=str(project_root / "data" / "chroma"),
)

# Retrieve the single passage most similar to the query.
# k=1 limits the result count; it does not enforce a relevance threshold.
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)


def retrieve_hint(query: str):

    """Retrieve the documentation passage most similar to the query.

            Args:
                query: Text describing the Python topic or operation to look up.

            Returns:
                A LangChain Document containing passage text and source metadata,
                or None if retrieval returns no documents.

            Note:
                Retrieval requests one result without a relevance threshold.
                The returned passage may not address the specific question.
    """
    documents = retriever.invoke(query)

    if not documents:
        return None

    return documents[0]
