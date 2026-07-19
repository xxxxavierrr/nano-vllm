from __future__ import annotations

import argparse
import json
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


OUTPUT_SHARD = "mtp.safetensors"
USER_AGENT = "nano-vllm-mtp-extractor/1.0"


def _url(endpoint: str, source: str, revision: str, filename: str) -> str:
    return "/".join(
        (
            endpoint.rstrip("/"),
            urllib.parse.quote(source, safe="/"),
            "resolve",
            urllib.parse.quote(revision, safe=""),
            urllib.parse.quote(filename, safe="/"),
        )
    )


def _read_url(
    url: str,
    token: str | None,
    byte_range: tuple[int, int] | None = None,
) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
            status = getattr(response, "status", None)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"failed to fetch {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch {url}: {exc.reason}") from exc
    if byte_range is not None:
        expected = byte_range[1] - byte_range[0] + 1
        if status != 206 or len(data) != expected:
            raise RuntimeError(
                f"invalid byte-range response for {url}: "
                f"status={status}, expected={expected}, actual={len(data)}"
            )
    return data


def _read_json(endpoint: str, source: str, revision: str, filename: str, token):
    return json.loads(_read_url(_url(endpoint, source, revision, filename), token))


def _read_safetensors_header(
    endpoint: str,
    source: str,
    revision: str,
    filename: str,
    token: str | None,
) -> tuple[int, dict]:
    shard_url = _url(endpoint, source, revision, filename)
    header_size = struct.unpack("<Q", _read_url(shard_url, token, (0, 7)))[0]
    if not 0 < header_size <= 100 * 1024 * 1024:
        raise RuntimeError(f"invalid safetensors header size in {filename}")
    header = _read_url(shard_url, token, (8, 7 + header_size))
    return header_size, json.loads(header)


def _build_output_header(tensors: list[tuple[str, str, dict]]) -> tuple[bytes, int]:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, _, metadata in tensors:
        start, end = metadata["data_offsets"]
        size = end - start
        header[name] = {
            "dtype": metadata["dtype"],
            "shape": metadata["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return encoded + b" " * ((-len(encoded)) % 8), offset


def extract(args: argparse.Namespace) -> dict:
    endpoint = args.endpoint.rstrip("/")
    token = args.token or os.environ.get("HF_TOKEN")
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_SHARD
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; pass --force to replace it")

    config = _read_json(endpoint, args.source, args.revision, "config.json", token)
    index = _read_json(
        endpoint,
        args.source,
        args.revision,
        "model.safetensors.index.json",
        token,
    )
    selected = {
        name: shard
        for name, shard in index.get("weight_map", {}).items()
        if name.startswith(args.prefix)
    }
    if not selected:
        raise RuntimeError(
            f"{args.source}@{args.revision} contains no {args.prefix!r} tensors"
        )

    shard_headers = {
        shard: _read_safetensors_header(
            endpoint, args.source, args.revision, shard, token
        )
        for shard in sorted(set(selected.values()))
    }
    tensors = []
    for name in sorted(selected):
        shard = selected[name]
        metadata = shard_headers[shard][1].get(name)
        if metadata is None:
            raise RuntimeError(f"{name} is absent from {shard}")
        if metadata.get("dtype") != "BF16":
            raise TypeError(
                f"MTP v1 requires BF16 tensors; {name} uses {metadata.get('dtype')}"
            )
        tensors.append((name, shard, metadata))

    output_header, payload_bytes = _build_output_header(tensors)
    temp_path = output_path.with_suffix(".safetensors.partial")
    if temp_path.exists():
        temp_path.unlink()
    try:
        with temp_path.open("wb") as output:
            output.write(struct.pack("<Q", len(output_header)))
            output.write(output_header)
            written = 0
            for index, (name, shard, metadata) in enumerate(tensors, start=1):
                source_header_size = shard_headers[shard][0]
                start, end = metadata["data_offsets"]
                absolute_start = 8 + source_header_size + start
                absolute_end = 8 + source_header_size + end - 1
                print(
                    f"[{index}/{len(tensors)}] {name}: {end - start:,} bytes",
                    flush=True,
                )
                data = _read_url(
                    _url(endpoint, args.source, args.revision, shard),
                    token,
                    (absolute_start, absolute_end),
                )
                output.write(data)
                written += len(data)
            output.flush()
            os.fsync(output.fileno())
        if written != payload_bytes:
            raise RuntimeError(
                f"payload size mismatch: expected {payload_bytes}, wrote {written}"
            )
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    extracted_index = {
        "metadata": {"total_size": payload_bytes},
        "weight_map": {name: OUTPUT_SHARD for name, _, _ in tensors},
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(extracted_index, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": args.source,
        "revision": args.revision,
        "prefix": args.prefix,
        "tensor_count": len(tensors),
        "payload_bytes": payload_bytes,
        "shards": sorted(shard_headers),
    }
    (output_dir / "mtp_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(tensors)} tensors ({payload_bytes:,} bytes) to {output_path}",
        flush=True,
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract BF16 MTP tensors from a sharded HF checkpoint."
    )
    parser.add_argument("--source", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prefix", default="mtp.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    )
    parser.add_argument("--token", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    extract(parse_args(argv))


if __name__ == "__main__":
    main()
