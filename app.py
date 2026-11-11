
import streamlit as st

from src.pdf_processor import extract_pdf_chunks
from src.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="Research Paper AI Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Paper AI Assistant")
st.caption(
    "Upload research papers, retrieve relevant passages, "
    "and ask questions using local AI."
)

with st.sidebar:
    st.header("Document Library")
    uploaded_files = st.file_uploader(
        "Upload PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
    )

    model_name = st.text_input(
        "Ollama model",
        value="llama3.2",
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=1,
        max_value=10,
        value=5,
    )

    process_documents = st.button(
        "Process Documents",
        type="primary",
        use_container_width=True,
    )

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if process_documents:
    if not uploaded_files:
        st.warning("Upload at least one PDF first.")
    else:
        try:
            all_chunks = []

            with st.spinner("Extracting document text..."):
                for uploaded_file in uploaded_files:
                    chunks = extract_pdf_chunks(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                    )
                    all_chunks.extend(chunks)

            if not all_chunks:
                st.error(
                    "No extractable text found. "
                    "Scanned PDFs may require OCR."
                )
            else:
                with st.spinner("Building semantic search index..."):
                    pipeline = RAGPipeline(model_name=model_name)
                    pipeline.add_documents(all_chunks)

                st.session_state.pipeline = pipeline
                st.session_state.messages = []

                st.success(
                    f"Processed {len(uploaded_files)} PDF(s) "
                    f"and indexed {len(all_chunks)} passages."
                )

        except Exception as error:
            st.error(f"Document processing failed: {error}")

if st.session_state.pipeline is None:
    st.info("Upload your research papers and select Process Documents to begin.")
else:
    st.subheader("Ask your research question")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Retrieved sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['source']} — Page {source['page']}**"
                        )
                        st.caption(
                            f"Similarity score: {source['score']}"
                        )
                        st.write(source["text"])

    question = st.chat_input(
        "Ask something about your uploaded research papers..."
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating an answer..."):
                try:
                    result = st.session_state.pipeline.answer(
                        question,
                        top_k=top_k,
                    )

                    st.markdown(result["answer"])

                    with st.expander("Retrieved sources"):
                        for source in result["sources"]:
                            st.markdown(
                                f"**{source['source']} — Page {source['page']}**"
                            )
                            st.caption(
                                f"Similarity score: {source['score']}"
                            )
                            st.write(source["text"])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                    )

                except Exception as error:
                    error_message = (
                        f"Could not generate an answer: {error}"
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )
