"""
azure_search.py

Provides functionality to interact with Azure Cognitive Search by
submitting a vector search request and retrieving top chunks. Also contains
a function to call Azure OpenAI for final answer generation using LlamaIndex's
AzureOpenAI client.
"""

import logging
import requests
import json
import config

logger = logging.getLogger(__name__)


def query_azure_search(query_embedding):
    """
    Sends a vector search request to Azure Cognitive Search using the query_embedding.

    Args:
        query_embedding (List[float]): The embedding of the user's query.

    Returns:
        List[str]: A list of top chunk contents (strings) from the search results.

    Raises:
        ValueError: If the search request fails or if no relevant results are found.
    """
    try:
        headers = {
            'Content-Type': 'application/json',
            'api-key': config.SEARCH_ADMIN_KEY
        }
        search_payload = {
            "search": "*",
            "vectors": [{
                "value": query_embedding,
                "fields": "embedding",
                "k": 5
            }],
            "select": "content,metadata"
        }
        url = f"{config.SEARCH_ENDPOINT}/indexes/{config.SEARCH_INDEX_NAME}/docs/search?api-version=2023-07-01-Preview"
        response = requests.post(url, headers=headers, json=search_payload)

        if response.status_code != 200:
            logger.error(f"Azure Search query failed: {response.status_code} - {response.text}")
            raise ValueError(f"Azure Search query failed with status code {response.status_code}")

        search_results = response.json().get('value', [])
        if not search_results:
            logger.warning("No results retrieved from Azure Search.")
            return []

        top_chunks = [result.get('content') for result in search_results if result.get('content')]
        logger.info(f"Retrieved {len(top_chunks)} chunks from Azure Search.")
        return top_chunks

    except ValueError:
        # If we already raised a ValueError (e.g., non-200 status code), re-raise it unchanged
        raise

    except Exception as e:
        logger.error(f"Error querying Azure Search: {e}")
        # For network issues, JSON parse errors, etc., raise a generic error
        raise ValueError("Failed to query Azure Search.") from e


def call_azure_openai(client_query, top_chunks):
    """
    Uses top_chunks from Azure Search as context and calls Azure OpenAI
    via llama_index.llms.azure_openai.AzureOpenAI to generate a final answer.

    Args:
        client_query (str): The user's query.
        top_chunks (List[str]): The top document chunks retrieved from Azure Search.

    Returns:
        str: The final answer from Azure OpenAI.

    Raises:
        ValueError: If there's an error calling the LLM or generating a response.
    """
    try:
        from llama_index.llms.azure_openai import AzureOpenAI
        from llama_index.llms.openai.utils import ChatMessage

        client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_MODEL_VERSION,
            deployment_name=config.AZURE_OPENAI_DEPLOYMENT_NAME
        )

        # Prepare the context string from the top chunks
        context = "\n\n".join([f"Chunk: {chunk}" for chunk in top_chunks])

        # Build a list of ChatMessage objects
        messages = [
            ChatMessage(
                role="system",
                content="You are a helpful assistant. Use the provided context to answer the user's query accurately."
            ),
            ChatMessage(
                role="system",
                content=f"Context: {context}"
            ),
            ChatMessage(
                role="user",
                content=client_query
            ),
        ]

        response = client.chat(messages=messages, max_tokens=500, temperature=0)
        logger.info("Response generated successfully with Azure OpenAI.")

        # ChatResponse is an object, so we directly return response.message.content
        return response.message.content

    except Exception as e:
        logger.error(f"Error generating response with Azure OpenAI: {e}")
        raise ValueError("Failed to call Azure OpenAI for final answer.") from e
