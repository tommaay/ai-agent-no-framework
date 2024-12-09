from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from loguru import logger
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OpenAIError(Exception):
    """Custom exception for OpenAI-related errors."""

    pass


class AgentBase(ABC):
    """
    Base class for AI agents that defines the common interface.

    Attributes:
        name: The agent's identifier
        max_retries: Maximum number of retry attempts for API calls
        verbose: Whether to log detailed information
    """

    def __init__(self, name: str, max_retries: int = 2, verbose: bool = True) -> None:
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's main functionality."""
        pass

    def call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 150,
    ) -> Dict[str, str]:
        """
        Make an API call to OpenAI's chat completion endpoint.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Controls randomness in the response (0-1)
            max_tokens: Maximum tokens in the response

        Returns:
            The message content from OpenAI's response

        Raises:
            OpenAIError: If the API call fails after max retries
        """
        retries = 0
        while retries < self.max_retries:
            try:
                if self.verbose:
                    logger.info(f"[{self.name}] Sending message to OpenAI")
                    for message in messages:
                        logger.debug(f" {message['role']}: {message['content']}")

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if not response.choices:
                    raise OpenAIError("No choices in OpenAI response")

                reply = response.choices[0].message

                if self.verbose:
                    logger.info(f"[{self.name}] Received response: {reply}")

                return reply

            except Exception as e:
                retries += 1
                logger.error(
                    f"[{self.name}] Error calling OpenAI: {str(e)}. "
                    f"Attempt {retries}/{self.max_retries}"
                )

        raise OpenAIError(f"Failed to get response after {self.max_retries} retries")
