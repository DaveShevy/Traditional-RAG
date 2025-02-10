"""
test_azure_blob.py

Pytest tests for azure_blob.py, which implements a client for listing and retrieving PDF blobs from Azure Blob Storage.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from utils.azure_blob import AzureBlobClient
from io import BytesIO

@pytest.fixture
def mock_config(monkeypatch):
    """
    Patch config constants used in azure_blob so no real network calls happen.
    """
    monkeypatch.setattr("utils.azure_blob.config.AZURE_STORAGE_CONNECTION_STRING", "fake_connection_string")
    monkeypatch.setattr("utils.azure_blob.config.BLOB_CONTAINER_NAME", "fake_container")

#
# Tests for __init__
#
@patch("azure.storage.blob.BlobServiceClient.from_connection_string")
def test_init_success(mock_from_conn, mock_config, caplog):
    """
    If BlobServiceClient.from_connection_string succeeds,
    AzureBlobClient initializes with no error, logs success.
    """
    fake_blob_service = MagicMock()
    mock_from_conn.return_value = fake_blob_service

    with caplog.at_level(logging.INFO):
        client = AzureBlobClient()

    mock_from_conn.assert_called_once_with("fake_connection_string")
    assert client.container_name == "fake_container"

    info_logs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("AzureBlobClient initialized successfully." in msg for msg in info_logs)

@patch("azure.storage.blob.BlobServiceClient.from_connection_string", side_effect=Exception("Connection error"))
def test_init_failure(mock_from_conn, mock_config, caplog):
    """
    If from_connection_string raises an Exception, we log error and raise it.
    """
    with pytest.raises(Exception) as exc_info:
        AzureBlobClient()

    assert "Connection error" in str(exc_info.value)

    error_logs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert any("Failed to initialize AzureBlobClient" in msg for msg in error_logs)


#
# Tests for list_pdf_blobs
#
@patch.object(AzureBlobClient, "__init__", return_value=None)
def test_list_pdf_blobs_success(mock_init, mock_config):
    """
    If container_client.list_blobs() yields some blobs,
    we return only those with a .pdf extension.
    """
    client = AzureBlobClient()
    client.blob_service_client = MagicMock()
    client.container_name = "fake_container"

    fake_container_client = MagicMock()
    client.blob_service_client.get_container_client.return_value = fake_container_client

    # Instead of MagicMock(name="report.pdf"), set .name explicitly
    mock1 = MagicMock()
    mock1.name = "report.pdf"

    mock2 = MagicMock()
    mock2.name = "image.png"

    mock3 = MagicMock()
    mock3.name = "notes.pdf"

    blob_list = [mock1, mock2, mock3]
    fake_container_client.list_blobs.return_value = blob_list

    result = client.list_pdf_blobs()
    assert result == ["report.pdf", "notes.pdf"]
    fake_container_client.list_blobs.assert_called_once()

@patch.object(AzureBlobClient, "__init__", return_value=None)
def test_list_pdf_blobs_failure(mock_init, caplog):
    """
    If listing blobs fails, we log error and raise.
    """
    client = AzureBlobClient()
    client.blob_service_client = MagicMock()
    client.container_name = "fake_container"

    fake_container_client = MagicMock()
    client.blob_service_client.get_container_client.return_value = fake_container_client

    fake_container_client.list_blobs.side_effect = Exception("List error")

    with pytest.raises(Exception) as exc_info:
        client.list_pdf_blobs()

    assert "List error" in str(exc_info.value)
    error_logs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert any("Error listing PDF blobs:" in msg for msg in error_logs)


#
# Tests for get_pdf_stream
#
@patch.object(AzureBlobClient, "__init__", return_value=None)
def test_get_pdf_stream_success(mock_init):
    """
    If we download a PDF blob with some non-empty bytes, we return a BytesIO stream.
    """
    client = AzureBlobClient()
    client.blob_service_client = MagicMock()
    client.container_name = "fake_container"

    fake_container_client = MagicMock()
    client.blob_service_client.get_container_client.return_value = fake_container_client
    fake_blob_client = MagicMock()
    fake_container_client.get_blob_client.return_value = fake_blob_client

    # Simulate a successful download of b"PDF BYTES"
    fake_downloader = MagicMock()
    fake_downloader.readall.return_value = b"PDF BYTES"
    fake_blob_client.download_blob.return_value = fake_downloader

    stream = client.get_pdf_stream("example.pdf")
    assert isinstance(stream, BytesIO)
    assert stream.getvalue() == b"PDF BYTES"

    fake_container_client.get_blob_client.assert_called_once_with("example.pdf")
    fake_downloader.readall.assert_called_once()


@patch.object(AzureBlobClient, "__init__", return_value=None)
def test_get_pdf_stream_empty(mock_init):
    """
    If the PDF blob is empty or readall() returns no bytes,
    we raise ValueError about the blob being empty.
    """
    client = AzureBlobClient()
    client.blob_service_client = MagicMock()
    client.container_name = "fake_container"

    fake_container_client = MagicMock()
    client.blob_service_client.get_container_client.return_value = fake_container_client
    fake_blob_client = MagicMock()
    fake_container_client.get_blob_client.return_value = fake_blob_client

    fake_downloader = MagicMock()
    fake_downloader.readall.return_value = b""
    fake_blob_client.download_blob.return_value = fake_downloader

    with pytest.raises(ValueError) as exc_info:
        client.get_pdf_stream("empty.pdf")

    # Now we expect the final message to mention "Blob 'empty.pdf' is empty or could not be downloaded."
    assert "Blob 'empty.pdf' is empty or could not be downloaded." in str(exc_info.value)


@patch.object(AzureBlobClient, "__init__", return_value=None)
def test_get_pdf_stream_failure(mock_init, caplog):
    """
    If get_blob_client or download_blob raises an Exception,
    we log error and raise ValueError("Failed to retrieve blob '...'.")
    """
    client = AzureBlobClient()
    client.blob_service_client = MagicMock()
    client.container_name = "fake_container"

    fake_container_client = MagicMock()
    client.blob_service_client.get_container_client.return_value = fake_container_client
    fake_blob_client = MagicMock()
    fake_container_client.get_blob_client.return_value = fake_blob_client

    fake_blob_client.download_blob.side_effect = Exception("Download error")

    with pytest.raises(ValueError) as exc_info:
        client.get_pdf_stream("bad_blob.pdf")

    assert "Failed to retrieve blob 'bad_blob.pdf'." in str(exc_info.value)

    error_logs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert any("Error retrieving blob stream for bad_blob.pdf:" in msg for msg in error_logs)
