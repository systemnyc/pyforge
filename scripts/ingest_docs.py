"""Build and inspect the Python documentation vector index.

    Run from the project root:
        uv run python scripts/ingest_docs.py

    Reads the data structures documentation, splits it into overlapping
    chunks, embeds them, and stores them in data/chroma. Prints sample
    chunks, embedding details, and retrieval results for inspection.

"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load the UTF-8 documentation file relative to the project root.
source_path = Path(r"docs_corpus/tutorial/datastructures.txt")
text = source_path.read_text(encoding="utf-8")

# Split into chunks of up to 1,000 characters, with up to 200 characters
# of overlap to retain context across chunk boundaries.
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")

for index, chunk in enumerate(chunks[:8]):
    print(f"\n------Chunk {index}-------")
    print(chunk)

# Embed documentation locally on the CPU using the retrieval model.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Inspect one embedding's dimensions and sample values.
vector = embedding_model.embed_query(chunks[0])

print(f"Vector lenght: {len(vector)}")
print(f"First five numbers {vector[:5]}")

# Embed and store documentation chunks in the persistent python_docs collection.
# Stable IDs identify chunks, and source metadata records their origin.
vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    collection_name="python_docs",
    persist_directory="data/chroma",
    ids=[f"datastructures-{index}" for index in range(len(chunks))],
    metadatas=[{"source": str(source_path)} for _ in chunks],
)

print(f"Stored {len(chunks)} chunks in Chroma")

# Retrieve two passages for a manual check of the stored index.
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

results = retriever.invoke("how do I retrieve a value from a dictionary?")

for document in results:
    print("---Retrieved Passage ---")
    print(document.page_content)
    print(document.metadata)
