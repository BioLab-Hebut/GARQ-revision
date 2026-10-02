"""Low-overhead CPU/GPU resource monitoring for auditable experiment stages."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Iterator

import psutil
import torch


@dataclass
class StageRecord:
    stage: str
    wall_time_seconds: float
    cpu_rss_start_bytes: int
    cpu_rss_end_bytes: int
    cpu_rss_peak_bytes: int
    gpu_allocated_peak_bytes: int
    gpu_reserved_peak_bytes: int
    status: str
    error_type: str = ""
    error_message: str = ""


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.records: list[dict] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        process = psutil.Process()
        start_rss = process.memory_info().rss
        peak_rss = start_rss
        stop = threading.Event()

        def sample() -> None:
            nonlocal peak_rss
            while not stop.wait(self.interval_seconds):
                try:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                except psutil.Error:
                    return

        sampler = threading.Thread(target=sample, daemon=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        sampler.start()
        status = "PASS"
        error_type = ""
        error_message = ""
        try:
            yield
        except Exception as error:
            status = "FAIL"
            error_type = type(error).__name__
            error_message = str(error)
            raise
        finally:
            stop.set()
            sampler.join(timeout=1)
            end_rss = process.memory_info().rss
            peak_rss = max(peak_rss, end_rss)
            record = StageRecord(
                stage=name,
                wall_time_seconds=time.perf_counter() - started,
                cpu_rss_start_bytes=start_rss,
                cpu_rss_end_bytes=end_rss,
                cpu_rss_peak_bytes=peak_rss,
                gpu_allocated_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                status=status,
                error_type=error_type,
                error_message=error_message,
            )
            self.records.append(asdict(record))


def enforce_resource_floor(min_available_ram_gb: float, min_free_gpu_gb: float) -> dict:
    available_ram = psutil.virtual_memory().available
    evidence = {
        "available_ram_bytes": available_ram,
        "required_available_ram_bytes": int(min_available_ram_gb * 1024**3),
        "cuda_available": torch.cuda.is_available(),
    }
    if available_ram < min_available_ram_gb * 1024**3:
        raise RuntimeError(
            f"Resource preflight failed: {available_ram / 1024**3:.1f} GiB RAM available; "
            f"{min_available_ram_gb:.1f} GiB required"
        )
    if torch.cuda.is_available():
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        evidence.update({"free_gpu_bytes": free_gpu, "total_gpu_bytes": total_gpu, "required_free_gpu_bytes": int(min_free_gpu_gb * 1024**3)})
        if free_gpu < min_free_gpu_gb * 1024**3:
            raise RuntimeError(
                f"Resource preflight failed: {free_gpu / 1024**3:.1f} GiB GPU memory free; "
                f"{min_free_gpu_gb:.1f} GiB required"
            )
    elif min_free_gpu_gb > 0:
        raise RuntimeError("Resource preflight failed: CUDA is unavailable")
    return evidence
