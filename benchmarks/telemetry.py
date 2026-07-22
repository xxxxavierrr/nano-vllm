from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Awaitable, Callable


@dataclass(frozen=True, slots=True)
class GpuSample:
    timestamp_s: float
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    power_watts: float | None


Reader = Callable[[], Awaitable[GpuSample]]


async def read_nvidia_smi() -> GpuSample:
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"nvidia-smi exited {process.returncode}")
    lines = stdout.decode("utf-8").strip().splitlines()
    if not lines:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    first_gpu = lines[0]
    fields = [field.strip() for field in first_gpu.split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi output: {first_gpu!r}")
    power = None if fields[3] in {"[N/A]", "N/A"} else float(fields[3])
    return GpuSample(
        timestamp_s=perf_counter(),
        utilization_percent=float(fields[0]),
        memory_used_mib=float(fields[1]),
        memory_total_mib=float(fields[2]),
        power_watts=power,
    )


class GpuTelemetryMonitor:
    """Optional periodic GPU telemetry with explicit missing provenance."""

    def __init__(self, interval_s: float, reader: Reader = read_nvidia_smi):
        if interval_s <= 0:
            raise ValueError("telemetry interval must be positive")
        self.interval_s = interval_s
        self.reader = reader
        self.samples: list[GpuSample] = []
        self.error: str | None = None
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def __aenter__(self):
        self._task = asyncio.create_task(self._collect())
        return self

    async def __aexit__(self, *_):
        self._stop.set()
        if self._task is not None:
            await self._task

    async def _collect(self):
        while not self._stop.is_set():
            try:
                self.samples.append(await self.reader())
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except TimeoutError:
                pass

    def report(self) -> dict:
        if not self.samples:
            return {
                "source": "nvidia-smi",
                "interval_s": self.interval_s,
                "available": False,
                "missing_reason": self.error or "no samples collected",
                "sample_count": 0,
            }
        powers = [sample.power_watts for sample in self.samples if sample.power_watts is not None]
        return {
            "source": "nvidia-smi",
            "interval_s": self.interval_s,
            "available": True,
            "missing_reason": self.error,
            "sample_count": len(self.samples),
            "utilization_percent": {
                "mean": sum(item.utilization_percent for item in self.samples) / len(self.samples),
                "max": max(item.utilization_percent for item in self.samples),
            },
            "memory_used_mib": {
                "mean": sum(item.memory_used_mib for item in self.samples) / len(self.samples),
                "max": max(item.memory_used_mib for item in self.samples),
            },
            "memory_total_mib": self.samples[0].memory_total_mib,
            "power_watts": (
                {"mean": sum(powers) / len(powers), "max": max(powers)}
                if powers
                else None
            ),
            "samples": [asdict(sample) for sample in self.samples],
        }


def unavailable_gpu_telemetry(reason: str = "disabled") -> dict:
    return {
        "source": None,
        "interval_s": None,
        "available": False,
        "missing_reason": reason,
        "sample_count": 0,
    }
