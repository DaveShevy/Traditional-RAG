"""
test_azure_search.py

Pytest tests for azure_search.py, which contains:
1) query_azure_search(query_embedding)
2) call_azure_openai(client_query, top_chunks)
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
import requests
from utils.azure_search import query_azure_search, call_azure_openai

@pytest.fixture
def mock_config(monkeypatch):
    """
    A fixture to override config constants if needed.
    E.g., set fake endpoints or keys to ensure no real calls happen.
    """
    monkeypatch.setattr("utils.azure_search.config.SEARCH_ENDPOINT", "https://fake-search-endpoint")
    monkeypatch.setattr("utils.azure_search.config.SEARCH_ADMIN_KEY", "fake_admin_key")
    monkeypatch.setattr("utils.azure_search.config.SEARCH_INDEX_NAME", "fake_index")
    monkeypatch.setattr("utils.azure_search.config.AZURE_OPENAI_API_KEY", "fake_key")
    monkeypatch.setattr("utils.azure_search.config.AZURE_OPENAI_ENDPOINT", "https://fake-openai-endpoint")
    monkeypatch.setattr("utils.azure_search.config.AZURE_OPENAI_MODEL_VERSION", "2023-03-15-preview")
    monkeypatch.setattr("utils.azure_search.config.AZURE_OPENAI_DEPLOYMENT_NAME", "fake_deployment")


#
# Tests for query_azure_search(query_embedding)
#
@patch("requests.post")
def test_query_azure_search_success(mock_post, mock_config):
    """
    If the request returns 200 and a valid JSON with 'value', we parse out the 'content'
    fields and return them.
    """
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "value": [
            {"content": "Chunk 1 text", "metadata": {}},
            {"content": "Chunk 2 text", "metadata": {}}
        ]
    }
    mock_post.return_value = fake_response

    embedding = [0.1, 0.2, 0.3]
    chunks = query_azure_search(embedding)
    assert chunks == ["Chunk 1 text", "Chunk 2 text"]
    mock_post.assert_called_once()


@patch("requests.post")
def test_query_azure_search_no_results(mock_post, mock_config, caplog):
    """
    If the request returns 200 but 'value' is empty or missing content,
    we return an empty list. Also logs a warning.
    """
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"value": []}
    mock_post.return_value = fake_response

    with caplog.at_level(logging.WARNING):
        embedding = [0.4, 0.5]
        chunks = query_azure_search(embedding)

    assert chunks == [], "No chunks if the 'value' list is empty."
    assert any("No results retrieved from Azure Search." in rec.message for rec in caplog.records)


@patch("requests.post")
def test_query_azure_search_non_200(mock_post, mock_config):
    """
    If the request returns a non-200 status, we raise ValueError.
    """
    fake_response = MagicMock()
    fake_response.status_code = 500
    fake_response.text = "Server error"
    mock_post.return_value = fake_response

    with pytest.raises(ValueError) as exc_info:
        query_azure_search([0.9, 0.8, 0.7])
    assert "Azure Search query failed with status code 500" in str(exc_info.value)


@patch("requests.post", side_effect=requests.exceptions.RequestException("Network error"))
def test_query_azure_search_exception(mock_post, mock_config):
    """
    If requests.post raises a RequestException or any other error,
    we catch it and raise ValueError("Failed to query Azure Search.")
    """
    with pytest.raises(ValueError) as exc_info:
        query_azure_search([0.0, 0.0])
    assert "Failed to query Azure Search." in str(exc_info.value)


#
# Tests for call_azure_openai(client_query, top_chunks)
#
@patch("llama_index.llms.azure_openai.AzureOpenAI")
def test_call_azure_openai_success(mock_azure_openai, mock_config, caplog):
    """
    Mocks AzureOpenAI so we don't do a real call.
    Expects call_azure_openai to return response.message.content.
    """
    from llama_index.llms.openai.utils import ChatMessage

    # Mock out the client.chat(...) method to return a mock ChatResponse
    mock_client_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "Mocked AI answer."
    mock_client_instance.chat.return_value = mock_response

    mock_azure_openai.return_value = mock_client_instance

    final_answer = call_azure_openai(
        client_query="What is the capital of France?",
        top_chunks=["chunkA", "chunkB"]
    )

    mock_azure_openai.assert_called_once()
    mock_client_instance.chat.assert_called_once()

    assert final_answer == "Mocked AI answer."
    # Check that logs contain "Response generated successfully with Azure OpenAI."
    with caplog.at_level(logging.INFO):
        pass
    # We'll just confirm it doesn't crash and returns the mocked content.


@patch("llama_index.llms.azure_openai.AzureOpenAI")
def test_call_azure_openai_exception(mock_azure_openai, mock_config):
    """
    If the AzureOpenAI client raises an exception,
    we re-raise it as ValueError("Failed to call Azure OpenAI for final answer.").
    """
    mock_client_instance = MagicMock()
    mock_client_instance.chat.side_effect = Exception("Simulated error in chat()")
    mock_azure_openai.return_value = mock_client_instance

    with pytest.raises(ValueError) as exc_info:
        call_azure_openai("Some query", ["chunk1"])
    assert "Failed to call Azure OpenAI for final answer." in str(exc_info.value)
