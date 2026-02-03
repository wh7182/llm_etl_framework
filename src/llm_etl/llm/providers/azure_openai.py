"""Azure OpenAI provider with instructor integration and retry logic."""

import logging
import os
import random
import time
from typing import Type, TypeVar

import instructor
from openai import AzureOpenAI
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError
from pydantic import BaseModel

from llm_etl.llm.client import LLMClient


logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class AzureOpenAIProvider(LLMClient):
    """
    Production Azure OpenAI client with instructor for structured outputs.

    Features:
    - Automatic environment variable configuration
    - Robust rate limit handling with exponential backoff
    - Transient error retry logic
    - Structured output validation via instructor
    - Configurable timeouts and token limits
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 4096
    ):
        """
        Initialize Azure OpenAI provider.

        Args:
            endpoint: Azure OpenAI endpoint URL (reads from AZURE_OPENAI_ENDPOINT if None)
            api_key: Azure OpenAI API key (reads from AZURE_OPENAI_API_KEY if None)
            deployment: Deployment name (reads from AZURE_OPENAI_DEPLOYMENT if None)
            api_version: API version (reads from AZURE_OPENAI_API_VERSION if None, defaults to "2024-02-15-preview")
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response

        Raises:
            ValueError: If required configuration is missing
        """
        # Read from environment with fallbacks
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        self.timeout = timeout
        self.max_tokens = max_tokens

        # Validate required parameters
        missing = []
        if not self.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.deployment:
            missing.append("AZURE_OPENAI_DEPLOYMENT")

        if missing:
            raise ValueError(
                f"Missing required Azure OpenAI configuration: {', '.join(missing)}. "
                f"Please set environment variables or pass values to constructor."
            )

        # Initialize Azure OpenAI client
        azure_client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            timeout=self.timeout
        )

        # Patch with instructor for structured outputs
        self.client = instructor.from_openai(azure_client)

        logger.info(
            f"Initialized AzureOpenAIProvider: deployment={self.deployment}, "
            f"api_version={self.api_version}, timeout={self.timeout}s"
        )

    def complete(self, messages: list[dict], response_model: Type[T]) -> T:
        """
        Send messages to Azure OpenAI and return validated structured response.

        Handles:
        - Rate limits (429) with exponential backoff
        - Transient errors (500, 503) with exponential backoff
        - Response validation via instructor + Pydantic

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            response_model: Pydantic model class to validate response against

        Returns:
            Validated instance of response_model

        Raises:
            RateLimitError: If rate limits persist after all retries
            APIError: If API errors persist after all retries
            ValidationError: If response doesn't match schema (instructor handles this)
        """
        max_retries = 5
        base_delay = 1.0  # Start with 1 second

        for attempt in range(max_retries + 1):
            try:
                # Call Azure OpenAI with instructor for structured output
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    response_model=response_model,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )

                # Success! Return validated response
                return response

            except RateLimitError as e:
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    logger.warning(
                        f"Rate limit hit (429) on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {delay:.2f}s... Error: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Rate limit persisted after {max_retries + 1} attempts. Giving up."
                    )
                    raise

            except (APIError, APIConnectionError, APITimeoutError) as e:
                # Handle transient errors (500, 503, connection issues, timeouts)
                if attempt < max_retries and self._is_retryable_error(e):
                    # Exponential backoff with jitter
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    logger.warning(
                        f"Transient API error on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {delay:.2f}s... Error: {type(e).__name__}: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"API error persisted after {max_retries + 1} attempts or non-retryable. "
                        f"Error: {type(e).__name__}: {str(e)}"
                    )
                    raise

            except Exception as e:
                # Non-retryable errors (validation errors, etc.) - let them propagate
                logger.error(f"Non-retryable error in Azure OpenAI call: {type(e).__name__}: {str(e)}")
                raise

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: Exception from OpenAI API

        Returns:
            True if error is retryable (500, 503, connection issues)
        """
        # APIError includes 500, 503, etc.
        if isinstance(error, APIError):
            # Check status code if available
            if hasattr(error, 'status_code'):
                return error.status_code in [500, 502, 503, 504]
            # If no status code, assume retryable
            return True

        # Connection errors and timeouts are retryable
        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return True

        return False
