
import requests


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/chat"

    def generate(self, question: str, context: str) -> str:
        prompt = f"""
You are a research paper assistant.

Answer the question using only the supplied document context.

Rules:
- Do not invent facts or references.
- If the context does not contain the answer, say so.
- Use clear academic language.
- Cite supporting passages using their source and page labels.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                },
            },
            timeout=180,
        )

        response.raise_for_status()
        data = response.json()

        return data["message"]["content"].strip()
