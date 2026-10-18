import itertools

import pytest

from gqa import gpu

CANNED = "NVIDIA GeForce RTX 3080, 10240, 2743, 31.52\n"


def test_parse_query_output():
    (r,) = gpu.parse_query_output(CANNED)
    assert r == gpu.GpuReading("NVIDIA GeForce RTX 3080", 10240, 2743, 31.52)


def test_parse_handles_missing_power_and_rejects_garbage():
    (r,) = gpu.parse_query_output("GPU, 100, 5, [N/A]\n")
    assert r.power_draw_w is None
    for bad in ["", "a, b\n", "GPU, x, 5, 1\n"]:
        with pytest.raises(gpu.GpuError):
            gpu.parse_query_output(bad)


def test_query_and_report_use_runner():
    calls = []

    def runner(args):
        calls.append(args)
        return CANNED if any(a.startswith("--query-gpu") for a in args) else "TABLE"

    assert gpu.query_gpus(runner)[0].memory_total_mib == 10240
    assert gpu.nvidia_smi_report(runner) == "TABLE"
    assert calls[0][0] == "nvidia-smi"


def test_run_reports_missing_binary(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(gpu.subprocess, "run", boom)
    with pytest.raises(gpu.GpuError, match="not found"):
        gpu._run(["nvidia-smi"])


def test_sampler_records_peak_baseline_and_power():
    used = itertools.chain([1000, 3000, 6500, 4000], itertools.repeat(1200))
    power = itertools.chain([30.0, 200.0, 220.0], itertools.repeat(40.0))

    def query():
        return [gpu.GpuReading("g", 10240, next(used), next(power))]

    ticks = iter([0.0, 10.0])
    s = gpu.GpuSampler(interval_s=0.001, query=query, clock=lambda: next(ticks))
    s.start()
    import time

    time.sleep(0.05)
    result = s.stop()
    assert result.baseline_memory_used_mib == 1000
    assert result.peak_memory_used_mib == 6500
    assert result.duration_s == 10.0
    assert result.samples >= 3
    assert result.mean_power_w is not None and result.energy_kwh == pytest.approx(result.mean_power_w * 10 / 3.6e6)


def test_sampler_context_manager_and_errors():
    def query():
        return [gpu.GpuReading("g", 100, 10, None)]

    with gpu.GpuSampler(interval_s=0.001, query=query) as s:
        pass
    assert s.result.mean_power_w is None and s.result.energy_kwh is None

    def failing():
        raise gpu.GpuError("down")

    s = gpu.GpuSampler(interval_s=0.001, query=failing)
    s.start()
    with pytest.raises(gpu.GpuError, match="no GPU readings"):
        s.stop()
