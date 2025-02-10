"""
main.py

A Streamlit-based application that allows users to query vector-embedded PDF documents
through a Retrieval-Augmented Generation (RAG) pipeline. It uses Azure OpenAI for
embeddings and for generating final answers, and Azure Cognitive Search for retrieval.
"""

import streamlit as st
import openai
import logging

import config  # We'll rely on config.py for Azure OpenAI keys, search endpoint, etc.
from utils.agent_tools import rag_pipeline  # We'll reuse the RAG logic here

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """
    Entry point for the Streamlit app.

    Steps:
    1) Set Azure OpenAI credentials.
    2) Create a text input for the user's query.
    3) Call the RAG pipeline to get an answer and relevant chunks.
    4) Display the final answer and show relevant chunks in an expander.
    """
    # Set up Azure OpenAI credentials
    openai.api_type = "azure"
    openai.api_key = config.AZURE_OPENAI_API_KEY
    openai.api_base = config.AZURE_OPENAI_ENDPOINT
    openai.api_version = config.AZURE_OPENAI_MODEL_VERSION

    st.set_page_config(page_title="PDF RAG Demo", page_icon="🤖", layout="wide")
    st.title("PDF RAG Assistant")

    user_question = st.text_input("Ask a question about your PDF documents:")

    if user_question:
        try:
            # RAG pipeline returns (answer, chunks)
            final_answer, top_chunks = rag_pipeline(user_question)
            st.markdown("**Answer:**")
            st.write(final_answer)

            with st.expander("Relevant Chunks from the PDF"):
                for idx, chunk in enumerate(top_chunks, start=1):
                    st.markdown(f"**Chunk {idx}:** {chunk}")

        except Exception as e:
            logger.exception("Error processing RAG query.")
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
