"""
Unit tests for LLM error handling, retries and rate limiting.
"""

from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as google_exceptions
from langchain_core.documents import Document

from app.errors import (
    DEFAULT_RETRY_WAIT,
    MAX_RETRY_WAIT,
    LLMError,
    handle_llm_error,
    is_rate_limit_error,
    is_timeout_error,
    llm_retry,
    retry_wait,
    suggested_retry_wait,
)


def rate_limited(seconds: float) -> google_exceptions.ResourceExhausted:
    """A 429 shaped like Gemini's, with a suggested retry delay."""
    return google_exceptions.ResourceExhausted(
        f"You exceeded your current quota. Please retry in {seconds}s."
    )


class TestRetryWait:
    """Tests for deciding whether and how long to wait before retrying."""

    @pytest.mark.unit
    def test_detects_rate_limit(self):
        assert is_rate_limit_error(rate_limited(1))
        assert is_rate_limit_error(Exception("429 You exceeded your current quota"))
        assert not is_rate_limit_error(ValueError("bad input"))

    @pytest.mark.unit
    def test_parses_suggested_wait(self):
        assert suggested_retry_wait(rate_limited(37.78)) == pytest.approx(37.78)
        assert suggested_retry_wait(Exception("retry_delay { seconds: 12 }")) == 12
        assert suggested_retry_wait(Exception("no hint here")) is None

    @pytest.mark.unit
    def test_waits_for_suggested_delay_within_cap(self):
        assert retry_wait(rate_limited(MAX_RETRY_WAIT)) == MAX_RETRY_WAIT

    @pytest.mark.unit
    def test_fails_fast_when_suggested_delay_exceeds_cap(self):
        assert retry_wait(rate_limited(MAX_RETRY_WAIT + 5)) is None

    @pytest.mark.unit
    def test_default_wait_without_hint(self):
        assert retry_wait(google_exceptions.ServiceUnavailable("busy")) == DEFAULT_RETRY_WAIT

    @pytest.mark.unit
    def test_does_not_retry_other_errors(self):
        assert retry_wait(ValueError("bad input")) is None

    @pytest.mark.unit
    def test_timeout_is_retried_like_503(self):
        timeout = google_exceptions.DeadlineExceeded("Deadline Exceeded")
        assert is_timeout_error(timeout)
        assert is_timeout_error(TimeoutError())
        assert retry_wait(timeout) == retry_wait(google_exceptions.ServiceUnavailable("busy"))


class TestLLMRetry:
    """Tests for the llm_retry decorator."""

    @pytest.mark.unit
    def test_retries_once_then_succeeds(self):
        calls = MagicMock(side_effect=[rate_limited(0.01), "ok"])

        result = llm_retry(calls)()

        assert result == "ok"
        assert calls.call_count == 2

    @pytest.mark.unit
    def test_gives_up_after_one_retry(self):
        calls = MagicMock(side_effect=rate_limited(0.01))

        with pytest.raises(google_exceptions.ResourceExhausted):
            llm_retry(calls)()

        assert calls.call_count == 2

    @pytest.mark.unit
    def test_does_not_retry_when_wait_too_long(self):
        calls = MagicMock(side_effect=rate_limited(MAX_RETRY_WAIT + 5))

        with pytest.raises(google_exceptions.ResourceExhausted):
            llm_retry(calls)()

        assert calls.call_count == 1

    @pytest.mark.unit
    @patch("app.errors.DEFAULT_RETRY_WAIT", 0.01)
    def test_retries_timeout_once_then_gives_up(self):
        calls = MagicMock(side_effect=google_exceptions.DeadlineExceeded("Deadline Exceeded"))

        with pytest.raises(google_exceptions.DeadlineExceeded):
            llm_retry(calls)()

        assert calls.call_count == 2

    @pytest.mark.unit
    def test_does_not_retry_non_retryable(self):
        calls = MagicMock(side_effect=ValueError("bad input"))

        with pytest.raises(ValueError):
            llm_retry(calls)()

        assert calls.call_count == 1


class TestHandleLLMError:
    """Tests for wrapping LLM failures in LLMError."""

    @pytest.mark.unit
    def test_rate_limit_message_includes_wait(self):
        @handle_llm_error
        def call():
            raise rate_limited(37.8)

        with pytest.raises(LLMError) as exc_info:
            call()

        err = exc_info.value
        assert err.details["rate_limited"] is True
        assert err.details["retry_after"] == pytest.approx(37.8)
        assert "rate limit" in err.message
        assert "38 seconds" in err.message

    @pytest.mark.unit
    def test_timeout_message(self):
        @handle_llm_error
        def call():
            raise google_exceptions.DeadlineExceeded("Deadline Exceeded")

        with pytest.raises(LLMError) as exc_info:
            call()

        err = exc_info.value
        assert err.details["timed_out"] is True
        assert err.details["rate_limited"] is False
        assert "did not respond within 30 seconds" in err.message

    @pytest.mark.unit
    def test_other_failure_message(self):
        @handle_llm_error
        def call():
            raise RuntimeError("boom")

        with pytest.raises(LLMError) as exc_info:
            call()

        assert exc_info.value.details["rate_limited"] is False
        assert "no answer could be produced" in exc_info.value.message
        assert "boom" not in exc_info.value.message


class TestStreamingRetry:
    """Tests for retrying streamed generation before the first token."""

    @staticmethod
    def chain_with_streams(*streams):
        """A mock chain whose successive astream() calls run the given async generators."""
        chain = MagicMock()
        chain.astream.side_effect = [s() for s in streams]
        return chain

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("app.chains.generator.get_generator_chain")
    async def test_retries_before_first_token(self, mock_get_chain):
        from app.chains.generator import generate_answer_stream

        async def failing():
            raise rate_limited(0.01)
            yield  # pragma: no cover

        async def working():
            for token in ["Hello ", "world"]:
                yield token

        mock_get_chain.return_value = self.chain_with_streams(failing, working)

        tokens = [t async for t in generate_answer_stream("q", [Document(page_content="c")])]

        assert tokens == ["Hello ", "world"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("app.chains.generator.get_generator_chain")
    async def test_raises_after_retry_fails(self, mock_get_chain):
        from app.chains.generator import generate_answer_stream

        async def failing():
            raise rate_limited(0.01)
            yield  # pragma: no cover

        mock_get_chain.return_value = self.chain_with_streams(failing, failing)

        with pytest.raises(LLMError) as exc_info:
            async for _ in generate_answer_stream("q", [Document(page_content="c")]):
                pass

        assert exc_info.value.details["rate_limited"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("app.chains.generator.get_generator_chain")
    async def test_raises_on_failure_mid_stream(self, mock_get_chain):
        from app.chains.generator import generate_answer_stream

        async def breaks():
            yield "Partial "
            raise RuntimeError("connection reset")

        mock_get_chain.return_value = self.chain_with_streams(breaks)

        received = []
        with pytest.raises(LLMError):
            async for token in generate_answer_stream("q", [Document(page_content="c")]):
                received.append(token)

        assert received == ["Partial "]
        assert mock_get_chain.return_value.astream.call_count == 1


class TestRateLimiter:
    """Tests for the shared client-side rate limiter."""

    @pytest.mark.unit
    def test_rate_limiter_is_shared_and_prefilled(self):
        from app.config import get_settings
        from app.llm import get_rate_limiter

        get_rate_limiter.cache_clear()
        limiter = get_rate_limiter()

        assert get_rate_limiter() is limiter
        assert limiter.requests_per_second == get_settings().llm_requests_per_second
        assert limiter.available_tokens == get_settings().llm_max_burst
        # A full bucket lets a 3-call query start without waiting
        assert all(limiter.acquire(blocking=False) for _ in range(3))
        get_rate_limiter.cache_clear()

    @pytest.mark.unit
    @patch("app.llm.TimeoutChatGoogleGenerativeAI")
    def test_get_llm_uses_shared_rate_limiter(self, mock_chat):
        from app.llm import get_llm, get_rate_limiter

        get_llm()

        assert mock_chat.call_args.kwargs["rate_limiter"] is get_rate_limiter()


class TestRequestTimeout:
    """Tests that the LLM client applies the timeout and owns no hidden retries."""

    @pytest.mark.unit
    def test_get_llm_sets_configured_timeout(self):
        from app.config import get_settings
        from app.llm import TimeoutChatGoogleGenerativeAI, get_llm

        llm = get_llm()

        assert isinstance(llm, TimeoutChatGoogleGenerativeAI)
        assert llm.timeout == get_settings().llm_timeout_seconds == 30.0

    @pytest.mark.unit
    def test_request_kwargs_add_timeout_and_disable_client_retry(self):
        from app.llm import get_llm

        kwargs = get_llm()._request_kwargs({})

        assert kwargs == {"timeout": 30.0, "retry": None}

    @pytest.mark.unit
    def test_timeout_reaches_client_and_is_attempted_once(self):
        """A timed-out request is sent once with the timeout; the library doesn't retry it."""
        from app.llm import get_llm, get_rate_limiter

        get_rate_limiter.cache_clear()
        llm = get_llm()
        timeout = google_exceptions.DeadlineExceeded("Deadline Exceeded")

        with patch.object(llm.client, "generate_content", side_effect=timeout) as mock_call:
            with pytest.raises(google_exceptions.DeadlineExceeded):
                llm.invoke("hello")

        mock_call.assert_called_once()
        assert mock_call.call_args.kwargs["timeout"] == 30.0
        assert mock_call.call_args.kwargs["retry"] is None
        get_rate_limiter.cache_clear()
