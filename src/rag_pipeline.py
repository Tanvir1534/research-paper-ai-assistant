
from src.embeddings import EmbeddingService
from src.llm import OllamaClient
from src.vector_store import VectorStore


class RAGPipeline:
    def __init__(self, model_name="llama3.2"):
        self.embeddings = EmbeddingService()
        self.llm = OllamaClient(model=model_name)
        self.vector_store = VectorStore(
            embedding_dimension=self.embeddings.model.get_sentence_embedding_dimension()
        )

    def add_documents(self, chunks):
        texts = [chunk.text for chunk in chunks]
        vectors = self.embeddings.encode_documents(texts)
        self.vector_store.add_documents(chunks, vectors)

    def answer(self, question: str, top_k: int = 5):
        query_vector = self.embeddings.encode_query(question)

        retrieved = self.vector_store.search(
            query_vector,
            top_k=top_k,
        )

        if not retrieved:
            return {
                "answer": "No relevant document passages were found.",
                "sources": [],
            }

        context_parts = []

        for item in retrieved:
            chunk = item["chunk"]
            context_parts.append(
                f"[Source: {chunk.source}, Page: {chunk.page}]\n"
                f"{chunk.text}"
            )

        context = "\n\n".join(context_parts)
        answer = self.llm.generate(question, context)

        sources = [
            {
                "source": item["chunk"].source,
                "page": item["chunk"].page,
                "score": round(item["score"], 4),
                "text": item["chunk"].text,
            }
            for item in retrieved
        ]

        return {
            "answer": answer,
            "sources": sources,
        }
