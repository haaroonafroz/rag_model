import faiss
import numpy as np

from typing import List, Tuple

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []  # Store chunks alongside the vectors

    def add_vectors(self, vectors: List[List[float]], chunks: List[str]) -> None:
        """Add vectors and their corresponding chunks to the vector store."""
        vectors_array = np.array(vectors).astype('float32')
        self.index.add(vectors_array)
        self.chunks.extend(chunks)

    def query(self, query_vector: List[float], top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Query the vector store for the most similar vectors."""
        query_array = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_array, top_k)
        
        # Debugging: Print the type and content of indices
        print(f"indices (raw): {indices}")

        # Get corresponding chunks
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        return retrieved_chunks, distances[0].tolist()        
    
    def get_total_chunks(self) -> int:
        """Return total number of chunks stored."""
        return len(self.chunks)
