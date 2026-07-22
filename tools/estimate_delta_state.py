from __future__ import annotations

import argparse
import json

from nanovllm.layers.fp8_delta_state import (
    DeltaStateShape,
    make_delta_state_layout,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare native and FP8 DeltaNet state capacity without a GPU"
    )
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--conv-channels", type=int, required=True)
    parser.add_argument("--conv-kernel-size", type=int, required=True)
    parser.add_argument("--recurrent-heads", type=int, required=True)
    parser.add_argument("--recurrent-key-dim", type=int, required=True)
    parser.add_argument("--recurrent-value-dim", type=int, required=True)
    parser.add_argument("--request-capacity", type=int, default=1)
    parser.add_argument("--branch-slots-per-request", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    shape = DeltaStateShape(
        layers=args.layers,
        conv_channels=args.conv_channels,
        conv_kernel_size=args.conv_kernel_size,
        recurrent_heads=args.recurrent_heads,
        recurrent_key_dim=args.recurrent_key_dim,
        recurrent_value_dim=args.recurrent_value_dim,
    )
    native = make_delta_state_layout(shape, dtype="auto")
    fp8 = make_delta_state_layout(shape, dtype="fp8_e4m3")
    result = {
        "shape": vars(args),
        "native": native.report(
            request_capacity=args.request_capacity,
            branch_slots_per_request=args.branch_slots_per_request,
        ),
        "fp8_e4m3": fp8.report(
            request_capacity=args.request_capacity,
            branch_slots_per_request=args.branch_slots_per_request,
        ),
        "compression_ratio": native.bytes_per_slot / fp8.bytes_per_slot,
        "runtime_enabled": False,
        "pending": "SM89 fused conv/recurrent numerical and Graph validation",
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
