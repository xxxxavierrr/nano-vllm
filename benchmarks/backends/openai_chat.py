import json
from time import perf_counter

import httpx

from benchmarks.models import ChatRequest, RequestResult


class OpenAIChatBackend:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        include_usage: bool = True,
    ):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/v1/chat/completions"
        self.model = model
        self.include_usage = include_usage

    async def validate_protocol(self, request: ChatRequest) -> None:
        models_response = await self.client.get(f"{self.base_url}/v1/models")
        models_response.raise_for_status()
        models = models_response.json().get("data", [])
        model_ids = {item.get("id") for item in models}
        if model_ids and self.model not in model_ids:
            raise RuntimeError(f"model {self.model!r} is not listed by /v1/models: {sorted(model_ids)}")

        payload = {
            "model": self.model,
            "messages": request.messages,
            "max_tokens": min(request.max_tokens, 4),
            "temperature": request.temperature,
            "stream": False,
        }
        response = await self.client.post(self.url, json=payload)
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", [])
        if not choices or "message" not in choices[0]:
            raise RuntimeError("non-streaming response has no assistant message")

    async def run(self, request: ChatRequest, scheduled_s: float) -> RequestResult:
        started_s = perf_counter()
        result = RequestResult(
            request_id=request.request_id,
            scheduled_s=scheduled_s,
            started_s=started_s,
            finished_s=started_s,
            session_id=request.session_id,
            turn_index=request.turn_index,
        )
        payload = {
            "model": self.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }
        if self.include_usage:
            payload["stream_options"] = {"include_usage": True}

        chunks = []
        try:
            async with self.client.stream("POST", self.url, json=payload) as response:
                result.status_code = response.status_code
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    result.error = f"HTTP {response.status_code}: {body}"
                    return result
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        result.saw_done = True
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        result.error = f"invalid SSE JSON: {exc}"
                        break
                    if "error" in event:
                        result.error = event["error"].get("message", "server error")
                        break
                    usage = event.get("usage")
                    if usage:
                        result.prompt_tokens = usage.get("prompt_tokens")
                        result.completion_tokens = usage.get("completion_tokens", 0)
                        details = usage.get("prompt_tokens_details") or {}
                        result.cached_tokens = details.get("cached_tokens")
                        result.token_count_source = "usage"
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason") is not None:
                        result.finish_reason = choice["finish_reason"]
                    content = choice.get("delta", {}).get("content")
                    if content:
                        now = perf_counter()
                        if result.first_content_s is None:
                            result.first_content_s = now
                        result.chunk_times_s.append(now)
                        chunks.append(content)
        except (httpx.HTTPError, OSError, TimeoutError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.finished_s = perf_counter()
            result.text = "".join(chunks)
            if result.token_count_source != "usage":
                result.completion_tokens = len(result.chunk_times_s)
                result.token_count_source = "sse_chunks"
            if result.status_code == 200 and not result.saw_done and result.error is None:
                result.error = "stream ended without [DONE]"
        return result
