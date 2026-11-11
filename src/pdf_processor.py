
from dataclasses import dataclass
from io import BytesIO

import fitz


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int


def extract_pdf_chunks(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    chunks = []

    with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            text = " ".join(text.split())

            if not text:
                continue

            start = 0

            while start < len(text):
                end = min(start + chunk_size, len(text))
                content = text[start:end].strip()

                if content:
                    chunks.append(
                        DocumentChunk(
                            text=content,
                            source=filename,
                            page=page_number,
                        )
                    )

                if end == len(text):
                    break

                start = end - overlap

    return chunks
