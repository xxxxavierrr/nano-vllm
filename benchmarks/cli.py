import argparse


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Unified nano-vLLM offline and OpenAI-compatible online benchmark"
    )
    parser.add_argument("--mode", choices=("offline", "online"), required=True)
    args, runner_argv = parser.parse_known_args(argv)

    if args.mode == "online":
        from benchmarks.online import main as run_online

        return run_online(runner_argv)

    from benchmarks.offline import main as run_offline

    return run_offline(runner_argv)


if __name__ == "__main__":
    main()
