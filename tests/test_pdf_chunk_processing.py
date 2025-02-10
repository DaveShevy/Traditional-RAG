"""
test_pdf_chunk_processing.py

Pytest tests for pdf_chunk_processing.py, which provides:
1) extract_text_from_pdf_stream(pdf_stream)
2) The PDFChunker class for chunking & embedding with Azure OpenAI
   and LlamaIndex's SemanticSplitterNodeParser.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from io import BytesIO
from PyPDF2 import PdfWriter

from utils.pdf_chunk_processing import extract_text_from_pdf_stream, PDFChunker


@pytest.fixture
def sample_pdf_with_text() -> bytes:
    """
    Generate an in-memory PDF containing a single page.
    We also set some metadata 'Title=Hello World'.

    Note: PyPDF2 won't parse 'Hello World' as text from metadata,
    so the actual extracted text is likely empty. We mainly confirm
    that the PDF is valid (no corruption error).
    """
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_metadata({"/Title": "Hello World"})
    writer.write(buffer)
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def sample_pdf_no_text() -> bytes:
    """
    Generate an in-memory PDF with a single blank page and no metadata.
    Should also yield no text for extraction.
    """
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.write(buffer)
    buffer.seek(0)
    return buffer.read()


def test_extract_text_success(sample_pdf_with_text, caplog):
    """
    Test extract_text_from_pdf_stream() with a valid minimal PDF.
    Because there's no real embedded text objects, page.extract_text()
    will likely return empty. If it's empty, your code should raise
    ValueError("The PDF appears to have no extractable text.").
    We'll confirm that behavior or see if partial text is found.
    """
    pdf_stream = BytesIO(sample_pdf_with_text)
    with caplog.at_level(logging.WARNING):
        try:
            extracted_text = extract_text_from_pdf_stream(pdf_stream)
        except ValueError as e:
            # If it's empty, we expect a ValueError about no text.
            assert "no extractable text" in str(e).lower()
            return

    # If we reach here, no exception was raised, meaning the code
    # found some text. Let's confirm it's not empty:
    assert extracted_text.strip(), (
        "Expected some text or a ValueError if none is found."
    )

def test_extract_text_no_text(sample_pdf_no_text):
    """
    If the PDF truly has no text, the code raises:
    ValueError("The PDF appears to have no extractable text.")
    """
    pdf_stream = BytesIO(sample_pdf_no_text)
    with pytest.raises(ValueError) as exc_info:
        extract_text_from_pdf_stream(pdf_stream)
    assert "the pdf appears to have no extractable text" in str(exc_info.value).lower()


def test_extract_text_corrupted_pdf():
    """
    If the PDF is truly corrupted, your code raises
    ValueError("Could not process the PDF stream.").
    """
    corrupt_stream = BytesIO(b"GARBAGE_DATA_NOT_A_PDF")
    with pytest.raises(ValueError) as exc_info:
        extract_text_from_pdf_stream(corrupt_stream)
    assert "could not process the pdf stream" in str(exc_info.value).lower()


@patch("llama_index.embeddings.azure_openai.AzureOpenAIEmbedding.__init__", return_value=None)
@patch("llama_index.embeddings.azure_openai.AzureOpenAIEmbedding.get_text_embedding", return_value=[0.1, 0.2, 0.3])
@patch("llama_index.core.node_parser.SemanticSplitterNodeParser.from_defaults")
def test_pdfchunker_init_and_chunk_text(mock_splitter_init, mock_embed, mock_embed_init, caplog):
    """
    Test PDFChunker initialization + chunk_text success scenario.
    We patch:
      - The AzureOpenAIEmbedding.__init__ so we skip real network calls
      - The embedding's get_text_embedding to return dummy vectors
      - The semantic splitter so we control how many nodes are returned
    """
    # Mock: The semantic splitter returns 2 fake nodes with content
    fake_node1 = MagicMock()
    fake_node1.get_content.return_value = "Chunk A"
    fake_node2 = MagicMock()
    fake_node2.get_content.return_value = "Chunk B"

    mock_splitter_instance = MagicMock()
    mock_splitter_instance.get_nodes_from_documents.return_value = [fake_node1, fake_node2]
    mock_splitter_init.return_value = mock_splitter_instance

    chunker = PDFChunker()
    assert hasattr(chunker, "embed_model"), "Should have an embed_model"
    assert hasattr(chunker, "splitter"), "Should have a splitter"

    text_to_chunk = "This is sample text for chunking."
    with caplog.at_level(logging.INFO):
        chunks = chunker.chunk_text(text_to_chunk, filename="test.pdf")

    # The splitter was called once
    mock_splitter_instance.get_nodes_from_documents.assert_called_once()
    # The embed model was called for each node
    assert mock_embed.call_count == 2, "Should embed each chunk"

    assert len(chunks) == 2, "Expected 2 chunked nodes"
    for node in chunks:
        assert node.embedding == [0.1, 0.2, 0.3], "Each node gets a dummy embedding"

    # The code logs "Created X valid chunks for file: filename"
    info_logs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("Created 2 valid chunks for file: test.pdf" in msg for msg in info_logs), (
        "Should log creation of 2 valid chunks for file: test.pdf"
    )


def test_pdfchunker_empty_text():
    """
    If chunk_text is called with empty or whitespace text,
    the method returns an empty list and logs a warning.
    """
    chunker = PDFChunker()
    # We rely on environment variables or mocking for embed_model init if needed.

    with patch.object(chunker, "splitter", MagicMock()) as mock_splitter, \
         patch.object(chunker, "embed_model", MagicMock()) as mock_embed, \
         patch("logging.Logger.warning") as warn_log:

        chunks = chunker.chunk_text("   ", filename="blank.pdf")
        assert chunks == [], "No chunks should be returned for empty text"

        warn_log.assert_called_once()
        assert "Text is empty or invalid" in warn_log.call_args[0][0], (
            "Should log a warning about empty text"
        )

        mock_splitter.get_nodes_from_documents.assert_not_called()
        mock_embed.get_text_embedding.assert_not_called()
