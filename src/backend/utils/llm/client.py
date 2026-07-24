"""Shared, lazily-initialized OpenAI client for backend modules."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_client = None  # OpenAI | TrackedOpenAI — proxy is duck-typed


def get_openai_client():
    """Get or create the shared OpenAI client instance (lazy initialization).

    Reads `OPENAI_API_KEY` and the optional `OPENAI_BASE_URL` from the environment.
    Raises `RuntimeError` on first call if `OPENAI_API_KEY` is not set.
    """
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")

        from utils.llm.tracking import TrackedOpenAI

        openai_client = TrackedOpenAI(OpenAI(api_key=api_key, base_url=base_url if base_url else None))
    return openai_client
