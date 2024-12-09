"""Agent for sanitizing medical data by removing Protected Health Information (PHI)."""

from typing import TypedDict, Optional
from loguru import logger
from openai import OpenAIError
from agents.agent_base import AgentBase


class SanitizedResponse(TypedDict):
    """Response type for data sanitization containing the processed data and metadata."""

    sanitized_data: str
    original_length: int
    sanitized_length: int
    phi_detected: bool


class SanitizeDataTool(AgentBase):
    """Tool for sanitizing medical data by removing PHI using OpenAI's API.

    Inherits from AgentBase to leverage common agent functionality.
    """

    def __init__(
        self,
        max_retries: int = 3,
        verbose: bool = True,
        phi_types: Optional[list[str]] = None,
    ) -> None:
        """Initialize the sanitization tool.

        Args:
            max_retries: Maximum number of API retry attempts
            verbose: Whether to log detailed information
            phi_types: List of PHI types to specifically target for removal
        """
        super().__init__(
            name="sanitize_data_tool", max_retries=max_retries, verbose=verbose
        )
        self.phi_types = phi_types or [
            "names",
            "dates",
            "addresses",
            "phone numbers",
            "email addresses",
            "SSN",
            "medical record numbers",
            "account numbers",
            "biometric identifiers",
        ]

    async def execute(self, medical_data: str) -> SanitizedResponse:
        """Sanitize medical data by removing PHI.

        Args:
            medical_data: The medical data to sanitize

        Returns:
            SanitizedResponse containing the sanitized data and metadata

        Raises:
            ValueError: If input data is empty or invalid
            OpenAIError: If API call fails
        """
        # Input validation
        if not medical_data or not medical_data.strip():
            raise ValueError("Medical data cannot be empty")

        if len(medical_data) > 8000:  # Adjust limit based on your needs
            raise ValueError("Input data exceeds maximum length")

        # Construct the system prompt with specific PHI types
        phi_types_str = ", ".join(self.phi_types)
        system_prompt = (
            "You are an AI assistant specialized in sanitizing medical data. "
            f"Remove all Protected Health Information (PHI) including: {phi_types_str}. "
            "Replace removed PHI with appropriate placeholders (e.g., [NAME], [DATE]). "
            "Preserve the medical context and meaning while ensuring HIPAA compliance."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Sanitize the following medical data:\n\n{medical_data}",
            },
        ]

        try:
            if self.verbose:
                logger.info(
                    f"Processing {len(medical_data)} characters of medical data"
                )

            response = await self.call_openai(
                messages=messages,
                temperature=0.1,  # Lower temperature for more consistent output
                max_tokens=1000,
            )

            sanitized_data = response["content"]

            # Simple heuristic to detect if PHI was found and removed
            phi_detected = any(
                placeholder in sanitized_data
                for placeholder in ["[NAME]", "[DATE]", "[ADDRESS]", "[PHONE]"]
            )

            result = SanitizedResponse(
                sanitized_data=sanitized_data,
                original_length=len(medical_data),
                sanitized_length=len(sanitized_data),
                phi_detected=phi_detected,
            )

            if self.verbose:
                logger.info(
                    f"Sanitization complete. Original length: {result['original_length']}, "
                    f"Sanitized length: {result['sanitized_length']}, "
                    f"PHI detected: {result['phi_detected']}"
                )

            return result

        except OpenAIError as e:
            logger.error(f"OpenAI API error during data sanitization: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data sanitization: {str(e)}")
            raise
