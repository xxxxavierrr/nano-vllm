import asyncio

from fastapi.testclient import TestClient

from nanovllm.serve.api_server import (
    ServerSettings,
    _build_replica_engine_kwargs,
    _stream_chat_completion,
    create_app,
    parse_args,
)
from nanovllm.serve.engine import RequestQueueFullError
from nanovllm.serve.protocol import MessageType


class FakeEngineRequest:
    def __init__(self, events):
        self._events = events
        self.aborted = False
        self.replica_id = 0

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


def test_graph_cli_defaults_and_explicit_eager_flag(tmp_path):
    defaults = parse_args(["--model", str(tmp_path)])
    assert defaults.cudagraph_mode == "FULL_AND_PIECEWISE"
    assert defaults.piecewise_max_tokens == 512
    assert defaults.startup_timeout == 1200.0

    eager = parse_args([
        "--model",
        str(tmp_path),
        "--enforce-eager",
        "--cudagraph-mode",
        "PIECEWISE",
    ])
    assert eager.enforce_eager is True


def test_data_parallel_cli_device_mapping(tmp_path):
    distributed = parse_args([
        "--model",
        str(tmp_path),
        "--data-parallel-size",
        "2",
        "--device-ids",
        "0,1",
    ])
    assert distributed.data_parallel_size == 2
    assert distributed.device_ids == [0, 1]
    assert distributed.data_parallel_simulate is False

    simulated = parse_args([
        "--model",
        str(tmp_path),
        "--data-parallel-size",
        "2",
        "--data-parallel-simulate",
        "--device-ids",
        "0",
    ])
    assert simulated.device_ids == [0, 0]
    replicas = _build_replica_engine_kwargs(simulated, "test-shm")
    assert [replica["device_ids"] for replica in replicas] == [[0], [0]]
    assert [replica["master_port"] for replica in replicas] == [2333, 2334]
    assert [replica["shm_name"] for replica in replicas] == [
        "test-shm-dp0",
        "test-shm-dp1",
    ]
    assert [replica["gpu_memory_utilization"] for replica in replicas] == [0.45, 0.9]


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
    assert response.headers["x-nanovllm-dp-replica"] == "0"


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
    assert response.headers["x-nanovllm-dp-replica"] == "0"


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
