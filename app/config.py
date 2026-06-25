"""
Central configuration for Memora.
Loads settings from the .env file and exposes them to the rest of the app.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()


class Settings:
    """Holds all project settings in one place."""

    # Qwen Cloud API credentials (read from .env)
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY")
    QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL")

    # Model names we'll use (from your free quota)
    LLM_MODEL: str = "qwen-plus-latest"          # main reasoning model
    LLM_MODEL_STRONG: str = "qwen3-max-2025-09-23"  # for harder tasks
    EMBEDDING_MODEL: str = "text-embedding-v4"   # converts text to vectors
    RERANK_MODEL: str = "qwen3-rerank"           # reorders search results

    def validate(self):
        """Check that critical settings are present. Fail early if not."""
        if not self.QWEN_API_KEY:
            raise ValueError("QWEN_API_KEY is missing. Check your .env file.")
        if not self.QWEN_BASE_URL:
            raise ValueError("QWEN_BASE_URL is missing. Check your .env file.")


# Create one shared settings object the whole app imports
settings = Settings()