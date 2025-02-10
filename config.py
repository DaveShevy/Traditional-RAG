"""
config.py

Loads environment variables using python-dotenv (.env) and sets up
global constants for Azure OpenAI, Azure Storage, and Azure Cognitive Search.

Raises EnvironmentError if a required variable is missing.
"""

import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

def get_env_variable(var_name: str) -> str:
    """
    Retrieve an environment variable by name.
    Raise EnvironmentError if not set.

    Args:
        var_name (str): The environment variable key to load.

    Returns:
        str: The environment variable's value.
    """
    val = os.getenv(var_name)
    if not val:
        logger.error(f"Environment variable '{var_name}' is not set.")
        raise EnvironmentError(f"Env var '{var_name}' is not set.")
    return val

# Azure OpenAI credentials
try:
    AZURE_OPENAI_API_KEY = get_env_variable('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_ENDPOINT = get_env_variable('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_DEPLOYMENT_NAME = get_env_variable('AZURE_OPENAI_DEPLOYMENT_NAME')
    AZURE_OPENAI_MODEL_VERSION = get_env_variable('AZURE_OPENAI_MODEL_VERSION')
    AZURE_OPENAI_EMBEDDING_NAME = get_env_variable('AZURE_OPENAI_EMBEDDING_NAME')
except EnvironmentError as e:
    logger.critical(f"Critical Azure OpenAI config error: {e}")
    raise

# Azure Blob Storage
try:
    AZURE_STORAGE_CONNECTION_STRING = get_env_variable('AZURE_STORAGE_CONNECTION_STRING')
    BLOB_CONTAINER_NAME = get_env_variable('BLOB_CONTAINER_NAME')
except EnvironmentError as e:
    logger.critical(f"Critical Azure Blob config error: {e}")
    raise

# Azure Cognitive Search
try:
    SEARCH_ENDPOINT = get_env_variable('SEARCH_ENDPOINT')
    SEARCH_ADMIN_KEY = get_env_variable('SEARCH_ADMIN_KEY')
    SEARCH_INDEX_NAME = get_env_variable('SEARCH_INDEX_NAME')
except EnvironmentError as e:
    logger.critical(f"Critical Azure Search config error: {e}")
    raise
