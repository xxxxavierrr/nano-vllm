import random

from benchmarks.models import ChatRequest


def _words(label: str, count: int) -> str:
    return " ".join(f"{label}{index % 97}" for index in range(max(0, count)))


def make_synthetic_requests(
    num_requests: int,
    input_len: int,
    output_len: int,
    shared_prefix_len: int,
    temperature: float,
    seed: int,
) -> list[ChatRequest]:
    if not 0 <= shared_prefix_len <= input_len:
        raise ValueError("shared_prefix_len must be between zero and input_len")
    rng = random.Random(seed)
    shared = _words("shared", shared_prefix_len)
    requests = []
    for index in range(num_requests):
        suffix_len = max(1, input_len - shared_prefix_len)
        suffix = _words(f"r{index}_", suffix_len)
        messages = []
        if shared:
            messages.append({"role": "system", "content": shared})
        messages.append({
            "role": "user",
            "content": f"{suffix}\nRequest nonce: {rng.randrange(2**32)}. Respond concisely.",
        })
        requests.append(ChatRequest(
            request_id=f"request-{index}",
            messages=messages,
            max_tokens=output_len,
            temperature=temperature,
        ))
    return requests
