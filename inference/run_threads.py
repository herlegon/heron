import time
from denc import (
    dlogger,
    BaseThread,
    DecoderThread,
    EncoderThread,
    ProgressThread,
)
from hytils import red
from hsys import VramMonitor


def _stop_all(threads: list[BaseThread], timeout: float = 5.0) -> list[BaseThread]:
    deadline = time.monotonic() + timeout
    alive = [t for t in threads if t.is_alive()]
    while alive and time.monotonic() < deadline:
        for t in alive:
            t.stop(force=True)
        time.sleep(0.05)
        alive = [t for t in alive if t.is_alive()]
    return alive



def run_threads(
    dec_thread: DecoderThread,
    infer_threads: list[BaseThread] | BaseThread,
    enc_thread: EncoderThread,
    progress_thread: ProgressThread | None = None,
    total_frames: int = 0,
    vram_stats: dict | None = None,
    vram_monitor: VramMonitor = None,
    debug: bool = False
) -> float:
    if not isinstance(infer_threads, (list, tuple)):
        infer_threads = [infer_threads]
    infer_threads = [t for t in infer_threads if t is not None]
    upstream: list[BaseThread] = [dec_thread, *infer_threads]
    chain: list[BaseThread] = [*upstream, enc_thread]

    for t in (dec_thread, enc_thread):
        t.verbose = debug
        t.is_cuda_workflow = True

    error: str | None = None
    start = time.perf_counter()
    elapsed = 0.0


    try:
        for t in chain:
            t.start()
        if not debug and progress_thread is not None:
            progress_thread.start()

        while enc_thread.is_alive():
            enc_thread.join(timeout=0.05)

            # Watchdog
            for t in upstream:
                if not t.is_alive():
                    failed, message = t.error_encountered()
                    if failed:
                        error = f"{t.name}: {message}"
                        break
            if error:
                break

        if error is None:
            failed, message = enc_thread.error_encountered()
            if failed:
                error = f"{enc_thread.name}: {message}"
            elif enc_thread.encoded != total_frames:
                error = f"{enc_thread.encoded}/{total_frames} frames encoded"

        # wait for the end of the encoding
        p = enc_thread.e_subprocess
        if p is not None:
            try:
                p.wait()
            except Exception as e:
                dlogger.error(f"wait ffmpeg: {e}")

        if progress_thread is not None:
            progress_thread.put(0, force=True)
        elapsed = time.perf_counter() - start

    finally:
        if vram_monitor is not None:
            vram_monitor.stop()
            if vram_stats is not None:
                vram_stats.update(vram_monitor.stats())

        stuck = _stop_all([dec_thread, *infer_threads], timeout=5.0)
        stuck += _stop_all([enc_thread], timeout=20.0)
        if progress_thread is not None:
            _stop_all([progress_thread], timeout=2.0)

            # In case of the progressbar stops below 100%:
            # deadline = time.monotonic() + 1.0
            # while (
            #     not progress_thread.in_queue.empty()
            #     and time.monotonic() < deadline
            # ):
            #     time.sleep(0.01)

        for t in stuck:
            dlogger.error(red(f"Error: {t.name} is still alive"))

    if error:
        dlogger.error(red(f"Error: {error}"))
        raise RuntimeError(error)


    return elapsed
