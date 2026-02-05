"""Unit tests for LLM client."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.exceptions import LLMConnectionError
from aria.llm.client import LLMClient, LLMResponse


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_creation(self) -> None:
        """Test creating LLMResponse."""
        response = LLMResponse(
            content="Hello, world!",
            model="claude-3-sonnet",
            input_tokens=10,
            output_tokens=5,
        )
        assert response.content == "Hello, world!"
        assert response.model == "claude-3-sonnet"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    def test_llm_response_with_optional_fields(self) -> None:
        """Test LLMResponse with optional fields."""
        response = LLMResponse(
            content="Response",
            model="claude-3-sonnet",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            metadata={"key": "value"},
        )
        assert response.stop_reason == "end_turn"
        assert response.metadata == {"key": "value"}

    def test_llm_response_default_optional_fields(self) -> None:
        """Test LLMResponse default values for optional fields."""
        response = LLMResponse(
            content="Test",
            model="model",
            input_tokens=1,
            output_tokens=1,
        )
        assert response.stop_reason is None
        assert response.metadata is None


class TestLLMClientInitialization:
    """Tests for LLMClient initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        with patch("aria.llm.client.AsyncAnthropic") as mock_anthropic:
            client = LLMClient(api_key="test-api-key")
            mock_anthropic.assert_called_once_with(api_key="test-api-key")
            assert client.model is not None

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        with patch("aria.llm.client.AsyncAnthropic"):
            client = LLMClient(model="claude-3-opus", api_key="test-key")
            assert client.model == "claude-3-opus"

    def test_init_without_api_key_raises_error(self) -> None:
        """Test initialization without API key raises LLMConnectionError."""
        with patch("aria.llm.client.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            mock_settings.anthropic_model = "claude-3-sonnet"

            with pytest.raises(LLMConnectionError) as exc_info:
                LLMClient()

            assert "API key not configured" in str(exc_info.value)

    def test_init_uses_settings_api_key(self) -> None:
        """Test initialization uses API key from settings."""
        with (
            patch("aria.llm.client.AsyncAnthropic") as mock_anthropic,
            patch("aria.llm.client.settings") as mock_settings,
        ):
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "settings-api-key"
            mock_settings.anthropic_api_key = mock_secret
            mock_settings.anthropic_model = "claude-3-sonnet"

            LLMClient()
            mock_anthropic.assert_called_once_with(api_key="settings-api-key")


class TestLLMClientComplete:
    """Tests for LLMClient.complete method."""

    @pytest.fixture
    def mock_client(self) -> LLMClient:
        """Create a client with mocked Anthropic."""
        with patch("aria.llm.client.AsyncAnthropic"):
            client = LLMClient(api_key="test-key", model="claude-3-sonnet")
        return client

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self, mock_client: LLMClient) -> None:
        """Test that complete returns LLMResponse."""

        @dataclass
        class MockContent:
            text: str = "Generated response"

        @dataclass
        class MockUsage:
            input_tokens: int = 10
            output_tokens: int = 20

        mock_response = MagicMock()
        mock_response.content = [MockContent()]
        mock_response.model = "claude-3-sonnet"
        mock_response.usage = MockUsage()
        mock_response.stop_reason = "end_turn"

        mock_client.client.messages.create = AsyncMock(return_value=mock_response)

        result = await mock_client.complete("Hello")

        assert isinstance(result, LLMResponse)
        assert result.content == "Generated response"
        assert result.input_tokens == 10
        assert result.output_tokens == 20

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self, mock_client: LLMClient) -> None:
        """Test complete with system prompt."""

        @dataclass
        class MockContent:
            text: str = "Response"

        @dataclass
        class MockUsage:
            input_tokens: int = 5
            output_tokens: int = 10

        mock_response = MagicMock()
        mock_response.content = [MockContent()]
        mock_response.model = "claude-3-sonnet"
        mock_response.usage = MockUsage()
        mock_response.stop_reason = "end_turn"

        mock_client.client.messages.create = AsyncMock(return_value=mock_response)

        await mock_client.complete("Hello", system="You are helpful")

        call_kwargs = mock_client.client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    @pytest.mark.asyncio
    async def test_complete_with_stop_sequences(self, mock_client: LLMClient) -> None:
        """Test complete with stop sequences."""

        @dataclass
        class MockContent:
            text: str = "Response"

        @dataclass
        class MockUsage:
            input_tokens: int = 5
            output_tokens: int = 10

        mock_response = MagicMock()
        mock_response.content = [MockContent()]
        mock_response.model = "claude-3-sonnet"
        mock_response.usage = MockUsage()
        mock_response.stop_reason = "stop_sequence"

        mock_client.client.messages.create = AsyncMock(return_value=mock_response)

        await mock_client.complete("Hello", stop_sequences=["END"])

        call_kwargs = mock_client.client.messages.create.call_args[1]
        assert call_kwargs["stop_sequences"] == ["END"]

    @pytest.mark.asyncio
    async def test_complete_empty_response_raises_error(
        self, mock_client: LLMClient
    ) -> None:
        """Test that empty response raises LLMResponseError."""
        mock_response = MagicMock()
        mock_response.content = []  # Empty content

        mock_client.client.messages.create = AsyncMock(return_value=mock_response)

        with pytest.raises(LLMConnectionError):
            await mock_client.complete("Hello")

    @pytest.mark.asyncio
    async def test_complete_api_error_raises_connection_error(
        self, mock_client: LLMClient
    ) -> None:
        """Test that API errors are wrapped in LLMConnectionError."""
        mock_client.client.messages.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(LLMConnectionError) as exc_info:
            await mock_client.complete("Hello")

        assert "API Error" in str(exc_info.value)


class TestLLMClientStream:
    """Tests for LLMClient.stream method."""

    @pytest.fixture
    def mock_client(self) -> LLMClient:
        """Create a client with mocked Anthropic."""
        with patch("aria.llm.client.AsyncAnthropic"):
            client = LLMClient(api_key="test-key", model="claude-3-sonnet")
        return client

    @pytest.mark.asyncio
    async def test_stream_yields_text_chunks(self, mock_client: LLMClient) -> None:
        """Test that stream yields text chunks."""
        chunks = ["Hello", " ", "world", "!"]

        async def async_text_generator():
            for chunk in chunks:
                yield chunk

        mock_stream = MagicMock()
        mock_stream.text_stream = async_text_generator()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        mock_client.client.messages.stream = MagicMock(return_value=mock_stream)

        result_chunks = []
        async for chunk in mock_client.stream("Hello"):
            result_chunks.append(chunk)

        assert result_chunks == chunks

    @pytest.mark.asyncio
    async def test_stream_with_system_prompt(self, mock_client: LLMClient) -> None:
        """Test stream with system prompt."""

        async def async_text_generator():
            yield "Response"

        mock_stream = MagicMock()
        mock_stream.text_stream = async_text_generator()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        mock_client.client.messages.stream = MagicMock(return_value=mock_stream)

        async for _ in mock_client.stream("Hello", system="Be helpful"):
            pass

        call_kwargs = mock_client.client.messages.stream.call_args[1]
        assert call_kwargs["system"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_stream_api_error_raises_connection_error(
        self, mock_client: LLMClient
    ) -> None:
        """Test that stream API errors are wrapped."""
        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(side_effect=Exception("Stream Error"))

        mock_client.client.messages.stream = MagicMock(return_value=mock_stream)

        with pytest.raises(LLMConnectionError) as exc_info:
            async for _ in mock_client.stream("Hello"):
                pass

        assert "Stream Error" in str(exc_info.value)


class TestLLMClientChat:
    """Tests for LLMClient.chat method."""

    @pytest.fixture
    def mock_client(self) -> LLMClient:
        """Create a client with mocked Anthropic."""
        with patch("aria.llm.client.AsyncAnthropic"):
            client = LLMClient(api_key="test-key", model="claude-3-sonnet")
        return client

    @pytest.mark.asyncio
    async def test_chat_with_message_history(self, mock_client: LLMClient) -> None:
        """Test chat with message history."""

        @dataclass
        class MockContent:
            text: str = "Response to conversation"

        @dataclass
        class MockUsage:
            input_tokens: int = 50
            output_tokens: int = 25

        mock_response = MagicMock()
        mock_response.content = [MockContent()]
        mock_response.model = "claude-3-sonnet"
        mock_response.usage = MockUsage()
        mock_response.stop_reason = "end_turn"

        mock_client.client.messages.create = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        result = await mock_client.chat(messages)

        assert isinstance(result, LLMResponse)
        assert result.content == "Response to conversation"

        call_kwargs = mock_client.client.messages.create.call_args[1]
        assert call_kwargs["messages"] == messages

    @pytest.mark.asyncio
    async def test_chat_with_system_prompt(self, mock_client: LLMClient) -> None:
        """Test chat with system prompt."""

        @dataclass
        class MockContent:
            text: str = "Response"

        @dataclass
        class MockUsage:
            input_tokens: int = 10
            output_tokens: int = 10

        mock_response = MagicMock()
        mock_response.content = [MockContent()]
        mock_response.model = "claude-3-sonnet"
        mock_response.usage = MockUsage()
        mock_response.stop_reason = "end_turn"

        mock_client.client.messages.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        await mock_client.chat(messages, system="You are a helpful assistant")

        call_kwargs = mock_client.client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_chat_api_error_raises_connection_error(
        self, mock_client: LLMClient
    ) -> None:
        """Test that chat API errors are wrapped."""
        mock_client.client.messages.create = AsyncMock(
            side_effect=Exception("Chat Error")
        )

        with pytest.raises(LLMConnectionError) as exc_info:
            await mock_client.chat([{"role": "user", "content": "Hello"}])

        assert "Chat Error" in str(exc_info.value)


class TestLLMClientParameters:
    """Tests for LLMClient parameter handling."""

    @pytest.fixture
    def mock_client(self) -> LLMClient:
        """Create a client with mocked Anthropic."""
        with patch("aria.llm.client.AsyncAnthropic"):
            client = LLMClient(api_key="test-key", model="claude-3-sonnet")
        return client

    @pytest.mark.asyncio
    async def test_complete_default_parameters(self, mock_client: LLMClient) -> None:
        """Test that complete uses default parameters."""

        @dataclass
        class MockContent:
            text: str = "Response"

        @dataclass
        class MockUsage:
            input_tokens: int = 10
            output_tokens: int = 10

        mock_response = MagicMock()
        mock_response.content = [MockContent()]
        mock_response.model = "claude-3-sonnet"
        mock_response.usage = MockUsage()
        mock_response.stop_reason = "end_turn"

        mock_client.client.messages.create = AsyncMock(return_value=mock_response)

        await mock_client.complete("Hello")

        call_kwargs = mock_client.client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_complete_custom_parameters(self, mock_client: LLMClient) -> None:
        """Test that complete uses custom parameters."""

        @dataclass
        class MockContent:
            text: str = "Response"

        @dataclass
        class MockUsage:
            input_tokens: int = 10
            output_tokens: int = 10

        mock_response = MagicMock()
        mock_response.content = [MockContent()]
        mock_response.model = "claude-3-sonnet"
        mock_response.usage = MockUsage()
        mock_response.stop_reason = "end_turn"

        mock_client.client.messages.create = AsyncMock(return_value=mock_response)

        await mock_client.complete("Hello", max_tokens=1000, temperature=0.5)

        call_kwargs = mock_client.client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["temperature"] == 0.5
