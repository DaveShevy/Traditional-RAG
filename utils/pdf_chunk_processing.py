"""
pdf_chunk_processing.py

Contains logic to:
1) Extract text from PDF streams using PyPDF2.
2) Chunk text into semantically meaningful segments using LlamaIndex's SemanticSplitterNodeParser.
3) Embed each chunk with Azure OpenAI embeddings.
"""

import logging
import re
from PyPDF2 import PdfReader
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
import config

logger = logging.getLogger(__name__)

def extract_text_from_pdf_stream(pdf_stream):
    """
    Extracts text from a file-like PDF stream using PyPDF2.
    Raises ValueError if the PDF is empty or fails to be read.
    """
    import logging
    from PyPDF2 import PdfReader
    logger = logging.getLogger(__name__)

    try:
        reader = PdfReader(pdf_stream)
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    logger.warning(f"No text found on page {i+1}.")
            except Exception as page_error:
                logger.error(f"Error reading page {i+1}: {page_error}")

        # Distinguish empty text from other errors
        if not text.strip():
            # Raise a specific ValueError that your test looks for
            raise ValueError("The PDF appears to have no extractable text.")

        logger.info("Text successfully extracted from the PDF stream.")
        return text

    except ValueError as e:
        # If we've already raised "The PDF appears to have no extractable text."
        # or some other ValueError, re-raise it as-is.
        logger.error(f"Extraction error: {e}")
        raise

    except Exception as e:
        # This is truly an unexpected or corrupted-PDF error
        logger.error(f"Failed to extract text from PDF stream: {e}")
        raise ValueError("Could not process the PDF stream.") from e


class PDFChunker:
    """
    A class to handle PDF text chunking and embedding using Azure OpenAI
    and LlamaIndex's SemanticSplitterNodeParser for semantic chunking.
    """

    def __init__(self):
        """
        Initializes the PDFChunker with an Azure OpenAI embedding model
        and a semantic splitter.
        """
        try:
            # Azure OpenAI embeddings
            self.embed_model = AzureOpenAIEmbedding(
                model=config.AZURE_OPENAI_EMBEDDING_NAME,
                deployment_name=config.AZURE_OPENAI_EMBEDDING_NAME,
                api_key=config.AZURE_OPENAI_API_KEY,
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_version=config.AZURE_OPENAI_MODEL_VERSION,
            )

            # Semantic splitter for chunking
            self.splitter = SemanticSplitterNodeParser.from_defaults(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embed_model
            )
            logger.info("PDFChunker initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing PDFChunker: {e}")
            raise

    def chunk_text(self, text: str, filename: str = ""):
        """
        Splits the given text into semantically meaningful chunks,
        computes embeddings, and returns a list of nodes with embeddings.

        Args:
            text (str): The raw text to be chunked.
            filename (str, optional): The associated filename for logging/metadata.

        Returns:
            list: A list of Node objects with .embedding set.
        """
        try:
            if not text.strip():
                logger.warning(f"Text is empty or invalid for file: {filename}.")
                return []

            documents = [Document(text=text, metadata={"filename": filename})]
            nodes = self.splitter.get_nodes_from_documents(documents)
            valid_nodes = []

            for i, node in enumerate(nodes):
                try:
                    chunk_embedding = self.embed_model.get_text_embedding(node.get_content())
                    if chunk_embedding is not None:
                        node.embedding = chunk_embedding
                        valid_nodes.append(node)
                    else:
                        logger.warning(f"Null embedding for chunk {i} in file {filename}.")
                except Exception as emb_err:
                    logger.warning(f"Error embedding chunk {i} in file {filename}: {emb_err}")

            logger.info(f"Created {len(valid_nodes)} valid chunks for file: {filename}")
            return valid_nodes
        except Exception as e:
            logger.error(f"Error during chunking for file {filename}: {e}")
            raise ValueError("Failed to chunk PDF text.") from e
