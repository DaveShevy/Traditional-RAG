"""
azure_blob.py

Implements a client for interacting with Azure Blob Storage. Allows listing
of PDF blob files and retrieving a specified PDF as a stream of bytes.
"""

import logging
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import config

logger = logging.getLogger(__name__)

class AzureBlobClient:
    """
    A client to interact with Azure Blob Storage (optional).
    """

    def __init__(self):
        """
        Initializes the AzureBlobClient with the BlobServiceClient,
        reading connection info from config.py.
        """
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                config.AZURE_STORAGE_CONNECTION_STRING
            )
            self.container_name = config.BLOB_CONTAINER_NAME
            logger.info("AzureBlobClient initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize AzureBlobClient: {e}")
            raise

    def list_pdf_blobs(self):
        """
        Lists all PDF files in the configured Azure Blob Storage container.

        Returns:
            list: A list of PDF blob filenames in the container.

        Raises:
            Exception: If there's an error accessing the container or blobs.
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blobs = container_client.list_blobs()
            pdf_blobs = [blob.name for blob in blobs if blob.name.endswith('.pdf')]
            logger.info(f"Found {len(pdf_blobs)} PDF blobs in the container.")
            return pdf_blobs
        except Exception as e:
            logger.error(f"Error listing PDF blobs: {e}")
            raise

    def get_pdf_stream(self, blob_name: str):
        """
        Retrieves the binary stream of a specified PDF blob from Azure Storage.

        Args:
            blob_name (str): The name of the PDF blob to retrieve.

        Returns:
            BytesIO: A BytesIO stream of the PDF's contents.

        Raises:
            ValueError: If the blob is empty or cannot be downloaded.
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)
            stream_downloader = blob_client.download_blob()
            pdf_bytes = stream_downloader.readall()

            if not pdf_bytes:
                # Raise a ValueError with a specific message
                raise ValueError(f"Blob '{blob_name}' is empty or could not be downloaded.")

            pdf_stream = BytesIO(pdf_bytes)
            logger.info(f"Successfully retrieved stream for blob: {blob_name}")
            return pdf_stream

        except ValueError:
            # If we specifically raised "Blob '...' is empty...", re-raise as-is
            logger.error(f"Error retrieving blob stream for {blob_name}: Blob is empty.")
            raise

        except Exception as e:
            logger.error(f"Error retrieving blob stream for {blob_name}: {e}")
            raise ValueError(f"Failed to retrieve blob '{blob_name}'.") from e
