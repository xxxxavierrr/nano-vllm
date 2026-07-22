from __future__ import annotations

from collections.abc import Awaitable, Callable


def point_passes_slo(
    result: dict,
    *,
    min_attainment: float,
    max_error_rate: float,
) -> bool:
    metrics = result["metrics"]
    attainment = metrics["slo"].get("attainment")
    if attainment is None:
        raise ValueError("offered-load sweep requires at least one SLO threshold")
    return (
        attainment >= min_attainment
        and metrics["requests"]["error_rate"] <= max_error_rate
    )


async def run_offered_load_sweep(
    run_point: Callable[[float], Awaitable[dict]],
    *,
    start_rate: float,
    growth_factor: float,
    max_rate: float,
    refine_steps: int,
    min_attainment: float,
    max_error_rate: float,
) -> tuple[list[dict], dict | None]:
    if start_rate <= 0 or max_rate < start_rate:
        raise ValueError("invalid sweep rate range")
    if growth_factor <= 1:
        raise ValueError("growth_factor must be greater than one")
    if refine_steps < 0:
        raise ValueError("refine_steps cannot be negative")

    points: list[dict] = []
    last_pass: tuple[float, dict] | None = None
    first_fail: tuple[float, dict] | None = None
    rate = start_rate
    while True:
        result = await run_point(rate)
        result["sweep"] = {
            "offered_request_rate": rate,
            "passed": point_passes_slo(
                result,
                min_attainment=min_attainment,
                max_error_rate=max_error_rate,
            ),
        }
        points.append(result)
        if result["sweep"]["passed"]:
            last_pass = (rate, result)
        else:
            first_fail = (rate, result)
            break
        if rate >= max_rate:
            break
        rate = min(max_rate, rate * growth_factor)

    if last_pass is not None and first_fail is not None:
        low, _ = last_pass
        high, _ = first_fail
        for _ in range(refine_steps):
            rate = (low + high) / 2
            result = await run_point(rate)
            passed = point_passes_slo(
                result,
                min_attainment=min_attainment,
                max_error_rate=max_error_rate,
            )
            result["sweep"] = {
                "offered_request_rate": rate,
                "passed": passed,
            }
            points.append(result)
            if passed:
                low = rate
                last_pass = (rate, result)
            else:
                high = rate

    points.sort(key=lambda item: item["sweep"]["offered_request_rate"])
    return points, last_pass[1] if last_pass is not None else None
