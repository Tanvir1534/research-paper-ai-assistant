
# Research Paper AI Assistant

A Retrieval-Augmented Generation (RAG) application for searching
and querying academic research papers using semantic embeddings,
FAISS vector search, and locally hosted language models.

## Overview

Research Paper AI Assistant allows users to upload PDF research
papers, retrieve relevant passages, and ask questions about their
contents through a conversational interface.

The application uses a local Ollama model for answer generation
and Sentence Transformers for semantic document retrieval.

## Features

- Upload multiple PDF research papers.
- Extract text while preserving page references.
- Split documents into overlapping text chunks.
- Generate semantic embeddings using Sentence Transformers.
- Perform similarity search with FAISS.
- Generate context-grounded answers using Ollama.
- Display retrieved passages and source-page references.
- Maintain conversation history during a session.
- Run without a paid external LLM API.

## Architecture

```text
              PDF Documents
                    |
                    v
              PDF Processor
                    |
                    v
              Text Chunking
                    |
                    v
           Sentence Transformers
                    |
                    v
               FAISS Index
                    |
User Question ---> Embedding
                    |
                    v
              Similarity Search
                    |
                    v
             Retrieved Context
                    |
                    v
               Ollama LLM
                    |
                    v
             Answer + Sources
```

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Programming Language | Python |
| PDF Processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| Language Model | Ollama |
| Retrieval Method | Semantic Similarity Search |

## Project Structure

```text
research-paper-ai-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── llm.py
└── data/
    └── .gitkeep
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/research-paper-ai-assistant.git
cd research-paper-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download Ollama:

https://ollama.com/download

Pull the default model:

```bash
ollama pull llama3.2
```

Make sure the Ollama service is running.

### 5. Launch the application

```bash
streamlit run app.py
```

The application will open in your browser.

## How It Works

1. Users upload one or more research papers.
2. PyMuPDF extracts text and page information.
3. Extracted text is divided into overlapping chunks.
4. Sentence Transformers converts chunks into embeddings.
5. FAISS indexes the document vectors.
6. A user question is converted into an embedding.
7. FAISS retrieves the most relevant passages.
8. Ollama generates an answer using the retrieved context.
9. The interface displays the answer and retrieved sources.

## Current Limitations

- Scanned PDFs require OCR before text extraction.
- The vector index is stored in memory and rebuilt when documents are processed.
- The application does not currently support persistent document storage.
- Answer quality depends on the retrieved passages and the selected language model.
- Similarity scores are retrieval scores, not verified measures of factual correctness.
- Source passages should be checked against the original papers.

## Future Improvements

- Persistent vector storage.
- OCR support for scanned research papers.
- Hybrid keyword and semantic retrieval.
- Reranking with cross-encoder models.
- Document summarization.
- Research-paper comparison.
- User authentication and document management.
- Automated evaluation of retrieval and answer quality.

## Project Status

Initial implementation. Further testing and evaluation are required
before production use.

## License

MIT License
