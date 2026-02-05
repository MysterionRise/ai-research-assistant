"""Unified LLM client for ARIA."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import structlog
from anthropic import AsyncAnthropic

from aria.config.settings import settings
from aria.exceptions import LLMConnectionError, LLMResponseError

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None
    metadata: dict | None = None


class LLMClient:
    """Unified client for LLM interactions.

    Supports Claude models via Anthropic API with streaming.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize LLM client.

        Args:
            model: Model name (default: from settings).
            api_key: API key (default: from settings).
        """
        self.model = model or settings.anthropic_model

        api_key = api_key or (
            settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        )

        if not api_key:
            raise LLMConnectionError("anthropic", "API key not configured")

        self.client = AsyncAnthropic(api_key=api_key)

        logger.info("llm_client_initialized", model=self.model)

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.
            stop_sequences: Optional stop sequences.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMConnectionError: If API call fails.
            LLMResponseError: If response is invalid.
        """
        logger.info(
            "llm_completion_request",
            prompt_length=len(prompt),
            max_tokens=max_tokens,
        )

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system:
                kwargs["system"] = system

            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences

            response = await self.client.messages.create(**kwargs)

            if not response.content:
                raise LLMResponseError("Empty response from LLM")

            result = LLMResponse(
                content=response.content[0].text,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                stop_reason=response.stop_reason,
            )

            logger.info(
                "llm_completion_completed",
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

            return result

        except Exception as e:
            logger.error("llm_completion_failed", error=str(e))
            raise LLMConnectionError("anthropic", str(e)) from e

    async def stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream a completion.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Yields:
            Text chunks as they're generated.
        """
        logger.info(
            "llm_stream_request",
            prompt_length=len(prompt),
            max_tokens=max_tokens,
        )

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system:
                kwargs["system"] = system

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

            logger.info("llm_stream_completed")

        except Exception as e:
            logger.error("llm_stream_failed", error=str(e))
            raise LLMConnectionError("anthropic", str(e)) from e

    async def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Multi-turn chat completion.

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}.
            system: Optional system prompt.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with generated content.
        """
        logger.info(
            "llm_chat_request",
            message_count=len(messages),
            max_tokens=max_tokens,
        )

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }

            if system:
                kwargs["system"] = system

            response = await self.client.messages.create(**kwargs)

            result = LLMResponse(
                content=response.content[0].text,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                stop_reason=response.stop_reason,
            )

            logger.info(
                "llm_chat_completed",
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

            return result

        except Exception as e:
            logger.error("llm_chat_failed", error=str(e))
            raise LLMConnectionError("anthropic", str(e)) from e
