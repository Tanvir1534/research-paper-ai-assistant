
import faiss
import numpy as np

from src.pdf_processor import DocumentChunk


class VectorStore:
    def __init__(self, embedding_dimension: int):
        self.index = faiss.IndexFlatIP(embedding_dimension)
        self.chunks: list[DocumentChunk] = []

    def add_documents(self, chunks, embeddings):
        if len(chunks) == 0:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Embedding and document counts do not match.")

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k: int = 5):
        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        query = query.reshape(1, -1)

        count = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, count)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append(
                {
                    "chunk": self.chunks[index],
                    "score": float(score),
                }
            )

        return results
