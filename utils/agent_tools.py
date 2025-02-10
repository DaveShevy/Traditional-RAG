"""
agent_tools.py

Implements the rag_pipeline function, which:
1) Gets a user query embedding from PDFChunker's AzureOpenAIEmbedding.
2) Uses that embedding to query Azure Cognitive Search for relevant chunks.
3) Calls Azure OpenAI for a final answer using those chunks.
"""

import logging
from utils.pdf_chunk_processing import PDFChunker
from utils.azure_search import query_azure_search, call_azure_openai

logger = logging.getLogger(__name__)

def rag_pipeline(user_query: str):
    """
    1) Convert user_query to an embedding via PDFChunker's AzureOpenAIEmbedding
    2) Use that embedding to query Azure Cognitive Search for the top relevant chunks
    3) Call Azure OpenAI with those chunks as context to produce a final answer

    Returns: (final_answer: str, top_chunks: list[str])
    """
    logger.info("Starting RAG pipeline for PDF documents.")
    try:
        chunker = PDFChunker()
        query_embedding = chunker.embed_model.get_text_embedding(user_query)
        if not query_embedding:
            # We raise a ValueError here. The test expects the final exception
            # to show "Query embedding is empty or invalid."
            raise ValueError("Query embedding is empty or invalid.")

        top_chunks = query_azure_search(query_embedding)
        if not top_chunks:
            # The test expects to see a fallback message but an empty chunk list.
            return "No relevant information found in the PDF docs.", []

        final_response = call_azure_openai(user_query, top_chunks)
        if not final_response:
            # The test expects we keep the chunk list, not return [].
            return "No response generated. Please refine your query.", top_chunks

        return final_response, top_chunks

    except ValueError as ve:
        # If we raised a ValueError from within the pipeline logic,
        # preserve that message (the test wants "Query embedding is empty..." etc.)
        logger.error(f"Error in RAG pipeline: {ve}")
        raise ve

    except Exception as e:
        # For all other unexpected errors, unify as "Failed RAG pipeline."
        logger.error(f"Unexpected error in RAG pipeline: {e}")
        raise ValueError("Failed RAG pipeline.") from e
