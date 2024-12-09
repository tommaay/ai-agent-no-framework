from typing import Dict, TypedDict
from loguru import logger
from openai import OpenAIError
from .agent_base import AgentBase


class SummaryResponse(TypedDict):
    summary: str
    original_length: int
    summary_length: int


class SummarizeTool(AgentBase):
    """Tool for summarizing medical texts using OpenAI's API.

    Inherits from AgentBase to leverage common agent functionality.
    """

    def __init__(self, max_retries: int = 2, verbose: bool = True) -> None:
        super().__init__(
            name="summarize_tool", max_retries=max_retries, verbose=verbose
        )

    async def execute(self, prompt: str) -> SummaryResponse:
        """Summarize the given text.

        Args:
            prompt: The text to summarize

        Returns:
            SummaryResponse containing the summary and metadata

        Raises:
            ValueError: If prompt is empty or too long
            OpenAIError: If API call fails
        """
        # Input validation
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if len(prompt) > 10000:  # Adjust limit as needed
            raise ValueError("Prompt exceeds maximum length")

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that creates clear and accurate summaries. "
                "Focus on the main points while preserving key details. "
                "Be concise but ensure no critical information is lost.",
            },
            {
                "role": "user",
                "content": f"Please provide a concise summary of the following text:\n\n{prompt}",
            },
        ]

        try:
            summary = await self.call_openai(messages, max_tokens=300)

            return SummaryResponse(
                summary=summary,
                original_length=len(prompt),
                summary_length=len(summary),
            )

        except OpenAIError as e:
            logger.error(f"OpenAI API error while summarizing text: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while summarizing text: {str(e)}")
            raise
