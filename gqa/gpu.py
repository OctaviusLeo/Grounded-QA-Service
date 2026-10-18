"""GPU telemetry from ``nvidia-smi`` (Req 1.1, 1.3) and energy for the cost column (Req 8.2).

:class:`GpuSampler` polls the GPU on a background thread between ``start`` and ``stop`` and keeps
the peak memory in use and the mean power draw. ``nvidia-smi`` reports memory for the whole card,
so the peak includes anything else using the GPU (for example the Windows desktop).
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

QUERY = "name,memory.total,memory.used,power.draw"


class GpuError(RuntimeError):
    """nvidia-smi is missing or its output cannot be parsed."""


@dataclass(frozen=True)
class GpuReading:
    name: str
    memory_total_mib: int
    memory_used_mib: int
    power_draw_w: float | None


Runner = Callable[[list[str]], str]


def _run(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False, timeout=30)
    except FileNotFoundError as exc:
        raise GpuError("nvidia-smi not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GpuError("nvidia-smi did not answer within 30 s") from exc
    if proc.returncode != 0:
        raise GpuError(f"nvidia-smi failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def parse_query_output(text: str) -> list[GpuReading]:
    """Parse ``nvidia-smi --query-gpu=<QUERY> --format=csv,noheader,nounits``."""
    readings = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            raise GpuError(f"unexpected nvidia-smi line: {line!r}")
        name, total, used, power = parts
        try:
            power_w = None if power in ("[N/A]", "N/A", "[Not Supported]") else float(power)
            readings.append(GpuReading(name, int(float(total)), int(float(used)), power_w))
        except ValueError as exc:
            raise GpuError(f"unexpected nvidia-smi line: {line!r}") from exc
    if not readings:
        raise GpuError("nvidia-smi reported no GPU")
    return readings


def query_gpus(runner: Runner = _run) -> list[GpuReading]:
    return parse_query_output(runner(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"]))


def nvidia_smi_report(runner: Runner = _run) -> str:
    """The plain ``nvidia-smi`` table, recorded verbatim as evidence."""
    return runner(["nvidia-smi"])


@dataclass(frozen=True)
class SamplerResult:
    samples: int
    duration_s: float
    peak_memory_used_mib: int
    baseline_memory_used_mib: int
    mean_power_w: float | None

    @property
    def energy_kwh(self) -> float | None:
        if self.mean_power_w is None:
            return None
        return self.mean_power_w * self.duration_s / 3_600_000.0


class GpuSampler:
    """Poll GPU 0 every ``interval_s`` seconds on a daemon thread."""

    def __init__(self, interval_s: float = 0.2, query: Callable[[], list[GpuReading]] = query_gpus,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.interval_s = interval_s
        self._query = query
        self._clock = clock
        self._readings: list[GpuReading] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._t1 = 0.0
        self._errors: list[str] = []

    def _sample(self) -> None:
        try:
            self._readings.append(self._query()[0])
        except GpuError as exc:
            self._errors.append(str(exc))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> GpuSampler:
        self._sample()  # baseline before the work starts
        self._t0 = self._clock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> SamplerResult:
        self._t1 = self._clock()
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()  # final reading after the work ends
        if not self._readings:
            raise GpuError("no GPU readings: " + "; ".join(self._errors[-3:]))
        powers = [r.power_draw_w for r in self._readings if r.power_draw_w is not None]
        return SamplerResult(
            samples=len(self._readings),
            duration_s=self._t1 - self._t0,
            peak_memory_used_mib=max(r.memory_used_mib for r in self._readings),
            baseline_memory_used_mib=self._readings[0].memory_used_mib,
            mean_power_w=(sum(powers) / len(powers)) if powers else None,
        )

    def __enter__(self) -> GpuSampler:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        if not self._stop.is_set():
            self.result = self.stop()
