import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []  # Store chunks alongside the vectors

    def add_vectors(self, vectors, chunks):
        """Add vectors and their corresponding chunks to the vector store."""
        self.index.add(np.array(vectors))
        self.chunks.extend(chunks)

    def query(self, vector, top_k=5):
        """Query the vector store for the most similar vectors."""
        distances, indices = self.index.search(np.array([vector]), top_k)
        return indices, distances

    def get_chunk(self, index):
        """Retrieve the chunk corresponding to the given index."""
        return self.chunks[index]
