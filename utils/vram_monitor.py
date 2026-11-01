import os, threading, time, statistics
import pynvml


class VramMonitor:
    def __init__(self, interval: float = 0.05, device: int = 0, skip_first: int = 0):
        self.interval = interval
        self.skip_first = skip_first   # nb d'échantillons ignorés (warmup)
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.device = device

    def _read_mb(self) -> float:
        pid = os.getpid()
        try:
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                if p.pid == pid and p.usedGpuMemory is not None:
                    return p.usedGpuMemory / 1024**2
        except pynvml.NVMLError:
            pass
        # Fallback (WDDM, Docker/WSL...) : mémoire totale du GPU - baseline
        return pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024**2 - self.baseline

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self._read_mb())
            time.sleep(self.interval)

    def start(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        self.baseline = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024**2
        self._thread = threading.Thread(target=self._run, daemon=True, name="vram-monitor")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        pynvml.nvmlShutdown()

    def stats(self) -> dict:
        s = self.samples[self.skip_first:] or self.samples
        if not s:
            return {"mean_mb": 0.0, "peak_mb": 0.0}
        return {"mean_mb": statistics.mean(s), "peak_mb": max(s)}
