import argparse
import json
import sys
import urllib.error
import urllib.request


EXIT_COMMANDS = {"/exit", "/quit"}


def stream_reply(opener, base_url, model, messages, max_tokens, temperature):
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    chunks = []
    finish_reason = None
    with opener.open(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"].get("message", "server error"))
            choices = event.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            content = choice.get("delta", {}).get("content")
            if content:
                chunks.append(content)
                print(content, end="", flush=True)
    print()
    return "".join(chunks), finish_reason


def print_help():
    print("Commands:")
    print("  /clear  Clear the conversation history")
    print("  /think  Toggle Qwen3 thinking mode (off by default)")
    print("  /exit   Exit the client (also: /quit)")
    print("  /help   Show this help")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run a streaming, multi-turn nano-vLLM chat client"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--system-prompt")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode initially",
    )
    args = parser.parse_args()

    initial_messages = []
    if args.system_prompt:
        initial_messages.append({"role": "system", "content": args.system_prompt})
    messages = list(initial_messages)
    thinking_enabled = args.thinking
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    print(f"Connected to {args.model} at {args.base_url}")
    print("Enter a message. Use /help for commands. Qwen3 thinking mode is off by default.")
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not prompt:
            continue
        command = prompt.lower()
        if command in EXIT_COMMANDS:
            print("Bye.")
            return
        if command == "/clear":
            messages = list(initial_messages)
            print("Conversation history cleared.")
            continue
        if command == "/help":
            print_help()
            continue
        if command == "/think":
            thinking_enabled = not thinking_enabled
            state = "on" if thinking_enabled else "off"
            print(f"Thinking mode: {state}")
            continue

        request_prompt = prompt
        if not thinking_enabled and "/no_think" not in command and "/think" not in command:
            request_prompt = f"{prompt} /no_think"
        messages.append({"role": "user", "content": request_prompt})
        print("Assistant> ", end="", flush=True)
        try:
            reply, finish_reason = stream_reply(
                opener,
                args.base_url,
                args.model,
                messages,
                args.max_tokens,
                args.temperature,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"\nHTTP {exc.code}: {detail}", file=sys.stderr)
            messages.pop()
            continue
        except urllib.error.URLError as exc:
            print(f"\nCannot connect to {args.base_url}: {exc.reason}", file=sys.stderr)
            messages.pop()
            continue
        except (ConnectionResetError, TimeoutError, OSError) as exc:
            print(f"\nConnection to {args.base_url} was interrupted: {exc}", file=sys.stderr)
            messages.pop()
            continue
        except KeyboardInterrupt:
            print("\nGeneration cancelled.")
            messages.pop()
            continue
        except (RuntimeError, json.JSONDecodeError) as exc:
            print(f"\nServer response error: {exc}", file=sys.stderr)
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        if finish_reason == "length":
            print(
                f"[Output reached max_tokens={args.max_tokens}. "
                "Use a larger -MaxTokens value or add /no_think to the prompt.]"
            )


if __name__ == "__main__":
    main()
