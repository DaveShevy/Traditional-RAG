"""
test_config.py

Pytest tests for config.py to ensure all required variables
cause an EnvironmentError if missing. Mocks load_dotenv() so
that local .env files do not interfere.
"""

import os
import sys
import pytest
import logging
import importlib
from unittest.mock import patch

REQUIRED_ENV_VARS = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_MODEL_VERSION",
    "AZURE_OPENAI_EMBEDDING_NAME",
    "AZURE_STORAGE_CONNECTION_STRING",
    "BLOB_CONTAINER_NAME",
    "SEARCH_ENDPOINT",
    "SEARCH_ADMIN_KEY",
    "SEARCH_INDEX_NAME",
]

@pytest.fixture
def clear_config_module():
    """
    Remove 'config' from sys.modules so we can re-import it
    and trigger the environment variable checks each time.
    """
    if "config" in sys.modules:
        del sys.modules["config"]
    yield
    if "config" in sys.modules:
        del sys.modules["config"]


@pytest.mark.usefixtures("clear_config_module")
@patch("dotenv.load_dotenv", return_value=None)  # prevents reading your actual .env
def test_all_vars_present(mock_load, caplog):
    """
    If all required environment variables are set, config.py
    should import with no CRITICAL logs or EnvironmentError.
    """
    full_env = {
        "AZURE_OPENAI_API_KEY": "some_key",
        "AZURE_OPENAI_ENDPOINT": "some_endpoint",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "some_deployment",
        "AZURE_OPENAI_MODEL_VERSION": "2023-03-15-preview",
        "AZURE_OPENAI_EMBEDDING_NAME": "some_embedding",
        "AZURE_STORAGE_CONNECTION_STRING": "blob_conn_str",
        "BLOB_CONTAINER_NAME": "some_blob_container",
        "SEARCH_ENDPOINT": "search_endpoint",
        "SEARCH_ADMIN_KEY": "search_key",
        "SEARCH_INDEX_NAME": "search_index",
    }

    with patch.dict(os.environ, full_env, clear=True):
        with caplog.at_level(logging.CRITICAL):
            import config
            importlib.reload(config)  # re-run top-level logic

        # Confirm no CRITICAL logs
        crit_logs = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert not crit_logs, "No CRITICAL logs expected if all vars are present."


@pytest.mark.usefixtures("clear_config_module")
@pytest.mark.parametrize("missing_var", REQUIRED_ENV_VARS)
@patch("dotenv.load_dotenv", return_value=None)  # prevents reading your actual .env
def test_missing_var_raises_env_error(mock_load, missing_var, caplog):
    """
    For each required var, remove it from the environment
    and confirm config.py raises EnvironmentError + logs CRITICAL.
    """
    # Provide everything except the missing var
    test_env = {}
    for var in REQUIRED_ENV_VARS:
        if var != missing_var:
            test_env[var] = f"some_value_for_{var}"

    with patch.dict(os.environ, test_env, clear=True):
        with caplog.at_level(logging.CRITICAL):
            with pytest.raises(EnvironmentError) as exc_info:
                import config
                importlib.reload(config)

        crit_logs = [r.message for r in caplog.records if r.levelno == logging.CRITICAL]
        assert crit_logs, f"Missing {missing_var} should produce CRITICAL log."

        combined_crit = "\n".join(crit_logs)
        exc_str = str(exc_info.value)
        # The missing var or relevant error message should appear
        assert missing_var in combined_crit or missing_var in exc_str, (
            f"Missing var '{missing_var}' not in logs or exception text."
        )
