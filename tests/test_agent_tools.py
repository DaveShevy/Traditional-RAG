"""
test_agent_tools.py

Pytest tests for agent_tools.py, specifically the rag_pipeline function:
1) Creates a PDFChunker and obtains a query embedding.
2) Searches Azure for relevant chunks.
3) Calls Azure OpenAI with those chunks to get a final answer.
4) Handles edge cases like no embedding, no chunks, or exceptions.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from utils.agent_tools import rag_pipeline

@patch("utils.agent_tools.call_azure_openai")
@patch("utils.agent_tools.query_azure_search")
@patch("utils.agent_tools.PDFChunker")
def test_rag_pipeline_success(mock_pdfchunker, mock_query_azure_search, mock_call_azure_openai):
    """
    Happy path:
    - The embedding is non-empty.
    - The search returns chunks.
    - call_azure_openai returns a final answer.
    """
    chunker_instance = MagicMock()
    # Suppose the embedding is [0.1, 0.2]
    chunker_instance.embed_model.get_text_embedding.return_value = [0.1, 0.2]
    mock_pdfchunker.return_value = chunker_instance

    # Suppose the search returns two chunks
    mock_query_azure_search.return_value = ["chunk1", "chunk2"]

    # Suppose OpenAI returns a final answer
    mock_call_azure_openai.return_value = "Some final answer"

    answer, chunks = rag_pipeline("What is in the PDF?")

    # Validate calls
    mock_pdfchunker.assert_called_once()
    chunker_instance.embed_model.get_text_embedding.assert_called_once_with("What is in the PDF?")
    mock_query_azure_search.assert_called_once_with([0.1, 0.2])
    mock_call_azure_openai.assert_called_once_with("What is in the PDF?", ["chunk1", "chunk2"])

    # Validate results
    assert answer == "Some final answer"
    assert chunks == ["chunk1", "chunk2"]


@patch("utils.agent_tools.call_azure_openai")
@patch("utils.agent_tools.query_azure_search")
@patch("utils.agent_tools.PDFChunker")
def test_rag_pipeline_empty_embedding(mock_pdfchunker, mock_query_azure_search, mock_call_azure_openai):
    """
    If the embedding returned is None or empty, the pipeline should raise ValueError.
    """
    chunker_instance = MagicMock()
    # Return an empty embedding
    chunker_instance.embed_model.get_text_embedding.return_value = []
    mock_pdfchunker.return_value = chunker_instance

    with pytest.raises(ValueError) as exc_info:
        rag_pipeline("Something")

    assert "Query embedding is empty or invalid." in str(exc_info.value)
    # Confirm we never call search or openai if embedding is empty
    mock_query_azure_search.assert_not_called()
    mock_call_azure_openai.assert_not_called()


@patch("utils.agent_tools.call_azure_openai")
@patch("utils.agent_tools.query_azure_search")
@patch("utils.agent_tools.PDFChunker")
def test_rag_pipeline_no_chunks(mock_pdfchunker, mock_query_azure_search, mock_call_azure_openai):
    """
    If the search returns no chunks, we return a fallback message and empty list.
    """
    chunker_instance = MagicMock()
    # Non-empty embedding
    chunker_instance.embed_model.get_text_embedding.return_value = [0.5, 0.6]
    mock_pdfchunker.return_value = chunker_instance

    # Return empty chunk list from search
    mock_query_azure_search.return_value = []

    answer, chunks = rag_pipeline("Any question")

    assert answer == "No relevant information found in the PDF docs."
    assert chunks == []

    # We never call openai if we have zero chunks
    mock_call_azure_openai.assert_not_called()


@patch("utils.agent_tools.call_azure_openai")
@patch("utils.agent_tools.query_azure_search")
@patch("utils.agent_tools.PDFChunker")
def test_rag_pipeline_no_final_response(mock_pdfchunker, mock_query_azure_search, mock_call_azure_openai):
    """
    If call_azure_openai returns no final response, we return a fallback message.
    """
    chunker_instance = MagicMock()
    chunker_instance.embed_model.get_text_embedding.return_value = [0.9, 1.0]
    mock_pdfchunker.return_value = chunker_instance

    mock_query_azure_search.return_value = ["chunkA"]
    # Suppose we get an empty or None response
    mock_call_azure_openai.return_value = ""

    answer, chunks = rag_pipeline("Where are my docs?")
    assert answer == "No response generated. Please refine your query."
    assert chunks == ["chunkA"]  # We still return the chunk list

    mock_call_azure_openai.assert_called_once_with("Where are my docs?", ["chunkA"])


@patch("utils.agent_tools.call_azure_openai")
@patch("utils.agent_tools.query_azure_search")
@patch("utils.agent_tools.PDFChunker")
def test_rag_pipeline_exception(mock_pdfchunker, mock_query_azure_search, mock_call_azure_openai):
    """
    If any unexpected exception occurs (like a network failure),
    we raise ValueError("Failed RAG pipeline.").
    """
    chunker_instance = MagicMock()
    chunker_instance.embed_model.get_text_embedding.return_value = [0.1, 0.2]
    mock_pdfchunker.return_value = chunker_instance

    mock_query_azure_search.return_value = ["chunk1"]
    # Let's force an exception in call_azure_openai
    mock_call_azure_openai.side_effect = Exception("Simulated error in LLM")

    with pytest.raises(ValueError) as exc_info:
        rag_pipeline("Check error handling")

    assert "Failed RAG pipeline." in str(exc_info.value)
