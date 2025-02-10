"""
test_main.py

Tests for main.py, which contains a Streamlit-based application.
We mock out Streamlit methods and the rag_pipeline to ensure
we can test the logic without requiring user interaction or real Azure calls.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
import main

@pytest.fixture
def mock_streamlit(monkeypatch):
    """
    A fixture to mock essential Streamlit UI calls so we don't open a browser
    or require manual input. Returns a dict of mock objects.
    """
    mocks = {}

    # Mock the streamlit functions we use in main.py
    mocks['set_page_config'] = MagicMock()
    mocks['title'] = MagicMock()
    mocks['text_input'] = MagicMock(return_value="")  # default: user enters nothing
    mocks['markdown'] = MagicMock()
    mocks['write'] = MagicMock()
    mocks['expander'] = MagicMock()
    mocks['error'] = MagicMock()

    monkeypatch.setattr(main.st, "set_page_config", mocks['set_page_config'])
    monkeypatch.setattr(main.st, "title", mocks['title'])
    monkeypatch.setattr(main.st, "text_input", mocks['text_input'])
    monkeypatch.setattr(main.st, "markdown", mocks['markdown'])
    monkeypatch.setattr(main.st, "write", mocks['write'])
    monkeypatch.setattr(main.st, "expander", mocks['expander'])
    monkeypatch.setattr(main.st, "error", mocks['error'])

    return mocks

@pytest.fixture
def mock_rag_pipeline(monkeypatch):
    """
    A fixture to mock the rag_pipeline function so we don't hit real Azure resources.
    Returns a MagicMock that we can configure in tests.
    """
    pipeline_mock = MagicMock(return_value=("Sample final answer", ["chunk1", "chunk2"]))
    monkeypatch.setattr("main.rag_pipeline", pipeline_mock)
    return pipeline_mock

def test_main_no_input(mock_streamlit, mock_rag_pipeline):
    """
    If user_question is empty, rag_pipeline should NOT be called
    and no final answer is displayed.
    """
    # text_input returns an empty string by default in mock_streamlit
    main.main()

    # rag_pipeline should not be called
    mock_rag_pipeline.assert_not_called()

    # We should not see any "Answer:" markdown or a call to st.write
    mock_streamlit['markdown'].assert_not_called()
    mock_streamlit['write'].assert_not_called()

def test_main_with_input(mock_streamlit, mock_rag_pipeline):
    """
    If user_question is non-empty, rag_pipeline is called, and we display the final answer + chunks.
    """
    mock_streamlit['text_input'].return_value = "What is the capital of France?"
    # Provide a custom pipeline return
    mock_rag_pipeline.return_value = ("Paris is the capital of France", ["chunkA", "chunkB"])

    main.main()

    # rag_pipeline should be called once with the user input
    mock_rag_pipeline.assert_called_once_with("What is the capital of France?")

    # We should see "Answer:", then the final answer, and an expander for chunks
    mock_streamlit['markdown'].assert_any_call("**Answer:**")
    mock_streamlit['write'].assert_any_call("Paris is the capital of France")

    # The expander was created
    mock_streamlit['expander'].assert_called_once_with("Relevant Chunks from the PDF")

def test_main_with_exception(mock_streamlit, mock_rag_pipeline, caplog):
    """
    If rag_pipeline raises an exception, the app should log the exception
    and display st.error.
    """
    # Simulate the user providing a query
    mock_streamlit['text_input'].return_value = "Any question"
    # Force rag_pipeline to raise an exception
    mock_rag_pipeline.side_effect = Exception("Simulated pipeline failure")

    with caplog.at_level(logging.ERROR):
        main.main()

    # Should have one error log from the exception
    error_logs = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(error_logs) > 0
    assert "Error processing RAG query" in error_logs[0].message

    # st.error should be called with the exception message
    mock_streamlit['error'].assert_called_once()
    args, kwargs = mock_streamlit['error'].call_args
    assert "Simulated pipeline failure" in args[0]
