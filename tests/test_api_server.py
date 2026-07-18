import asyncio

from fastapi.testclient import TestClient

from nanovllm.serve.api_server import ServerSettings, _stream_chat_completion, create_app
from nanovllm.serve.engine import RequestQueueFullError
from nanovllm.serve.protocol import MessageType


class FakeEngineRequest:
    def __init__(self, events):
        self._events = events
        self.aborted = False

    async def events(self):
        for event in self._events:
            yield event

    async def abort(self):
        self.aborted = True


class FakeClient:
    def __init__(self, reject=False):
        self.reject = reject
        self.requests = []
        self.process_alive = True

    async def submit(self, prompt_token_ids, sampling_params):
        if self.reject:
            raise RequestQueueFullError("engine request queue is full")
        request = FakeEngineRequest([
            {"type": MessageType.TOKEN, "token_id": 10},
            {"type": MessageType.TOKEN, "token_id": 11},
            {"type": MessageType.FINISHED, "finish_reason": "length"},
        ])
        self.requests.append((prompt_token_ids, sampling_params, request))
        return request


class FakeRuntime:
    def __init__(self, reject=False):
        self.max_model_len = 128
        self.client = FakeClient(reject=reject)

    def encode_messages(self, messages):
        return [1, 2, 3]

    def decode(self, token_ids):
        return "".join({10: "hello", 11: " world"}[token_id] for token_id in token_ids)

    async def healthy(self):
        return True


def make_client(runtime=None):
    settings = ServerSettings(model="unused", served_model_name="test-model")
    return TestClient(create_app(settings, runtime=runtime or FakeRuntime()))


def test_non_streaming_chat_completion():
    with make_client() as client:
        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
        })

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "hello world"
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_streaming_chat_completion_uses_openai_sse_frames():
    with make_client() as client:
        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_completion_tokens": 2,
        })

    assert response.status_code == 200
    assert '"role":"assistant"' in response.text
    assert '"content":"hello"' in response.text
    assert '"content":" world"' in response.text
    assert '"finish_reason":"length"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_streaming_chat_completion_can_include_usage():
    with make_client() as client:
        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 2,
        })

    assert response.status_code == 200
    assert '"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2' in response.text
    assert '"prompt_tokens_details":{"cached_tokens":0}' in response.text


def test_queue_full_returns_429_before_stream_starts():
    with make_client(FakeRuntime(reject=True)) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

    assert response.status_code == 429


def test_unsupported_request_field_is_rejected():
    with make_client() as client:
        response = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.5,
        })

    assert response.status_code == 422


def test_closing_stream_aborts_engine_request():
    async def scenario():
        runtime = FakeRuntime()
        engine_request = FakeEngineRequest([])
        stream = _stream_chat_completion(runtime, engine_request, "id", 1, "test-model")
        await anext(stream)
        await stream.aclose()
        assert engine_request.aborted

    asyncio.run(scenario())

