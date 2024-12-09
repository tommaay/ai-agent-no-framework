"""Tool for summarizing text content using OpenAI's API with error handling and logging."""

from typing import TypedDict
from loguru import logger
from openai import OpenAIError
from agents.agent_base import AgentBase


class SummaryResponse(TypedDict):
    """Response type for medical text summarization containing the summary and metadata."""

    summary: str
    original_length: int
    summary_length: int
    medical_terms_identified: list[str]  # New field for tracking medical terminology


class SummarizeTool(AgentBase):
    """Tool for summarizing medical texts using OpenAI's API.

    Specializes in medical content including clinical notes, research papers,
    and healthcare documentation while preserving critical medical terminology.
    """

    def __init__(self, max_retries: int = 2, verbose: bool = True) -> None:
        super().__init__(
            name="medical_summarize_tool", max_retries=max_retries, verbose=verbose
        )

    async def execute(self, prompt: str) -> SummaryResponse:
        """Summarize medical text while preserving critical medical information.

        Args:
            prompt: The medical text to summarize

        Returns:
            SummaryResponse containing the summary, metadata, and identified medical terms

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
                "content": (
                    "You are a medical AI assistant specialized in summarizing healthcare content. "
                    "Maintain clinical accuracy and preserve all critical medical information including: "
                    "- Diagnoses, conditions, and symptoms\n"
                    "- Medications, dosages, and treatments\n"
                    "- Lab results and vital signs\n"
                    "- Patient history and risk factors\n"
                    "Use precise medical terminology and maintain a professional clinical tone. "
                    "Also identify and list key medical terms used in the text."
                ),
            },
            {
                "role": "user",
                "content": f"Please provide a clinical summary of the following medical text, "
                f"followed by a list of key medical terms used:\n\n{prompt}",
            },
        ]

        try:
            response = await self.call_openai(messages, max_tokens=400)

            # Split the response into summary and medical terms
            parts = response.split("\nMedical terms:")
            summary = parts[0].strip()
            medical_terms = []
            if len(parts) > 1:
                medical_terms = [term.strip() for term in parts[1].split(",")]

            return SummaryResponse(
                summary=summary,
                original_length=len(prompt),
                summary_length=len(summary),
                medical_terms_identified=medical_terms,
            )

        except OpenAIError as e:
            logger.error(f"OpenAI API error while summarizing text: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while summarizing text: {str(e)}")
            raise
