from dataclasses import dataclass, field


@dataclass(slots=True)
class ChatRequest:
    request_id: str
    messages: list[dict[str, str]]
    max_tokens: int
    temperature: float
    session_id: str | None = None
    turn_index: int | None = None


@dataclass(slots=True)
class RequestResult:
    request_id: str
    scheduled_s: float
    started_s: float
    finished_s: float
    first_content_s: float | None = None
    chunk_times_s: list[float] = field(default_factory=list)
    status_code: int | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int = 0
    accepted_tokens: int | None = None
    cached_tokens: int | None = None
    token_count_source: str = "approximate"
    text: str = ""
    error: str | None = None
    saw_done: bool = False
    session_id: str | None = None
    turn_index: int | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.status_code == 200 and self.saw_done


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """Optional implementation-neutral engine telemetry sampled during a run."""

    timestamp_s: float
    running_requests: int
    scheduled_actual_tokens: int | None = None
    scheduled_padded_tokens: int | None = None
    accepted_tokens: int | None = None
