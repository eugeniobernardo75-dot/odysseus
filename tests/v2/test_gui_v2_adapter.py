import pytest

from routes.chat_routes import _form_flag_enabled, v2_chat_stream_adapter


class FakeUserAgent:
    def __init__(self, tokens):
        self.tokens = tokens
        self.call = None

    async def process_stream(self, message, context=None):
        self.call = (message, context)
        for token in self.tokens:
            yield token


@pytest.mark.asyncio
async def test_gui_v2_adapter_translates_tokens_and_done():
    context = {"model": "qwen3:8b"}
    user_agent = FakeUserAgent(["Olá", "!", "__DONE__"])

    events = [
        event async for event in v2_chat_stream_adapter(
            "Diz olá", context, user_agent=user_agent
        )
    ]

    assert events == [
        {"type": "metadata", "coordinated_by": "useragent"},
        {"type": "token", "content": "Olá"},
        {"type": "token", "content": "!"},
        {"type": "done"},
    ]
    assert user_agent.call == ("Diz olá", context)


@pytest.mark.asyncio
async def test_gui_v2_adapter_marks_missing_done_marker():
    events = [
        event async for event in v2_chat_stream_adapter(
            "Diz olá", user_agent=FakeUserAgent(["Olá"])
        )
    ]

    assert events[-1] == {"type": "incomplete"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), ("true", True), ("1", True), ("false", False), (None, False)],
)
def test_form_flag_enabled(value, expected):
    assert _form_flag_enabled(value) is expected
