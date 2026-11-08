from __future__ import annotations
from enum import IntEnum
from pprint import pprint
from queue import Queue
import time

import torch
from torch import Tensor
from hytils import blue, blue, red

from denc import (
    BaseThread,
    NnFrame,
    wait_frame_ready,
    mark_frame_ready,
    dlogger,
)
from nnarchs import (
    nnlib,
    Hdtype,
    PyTorchModel,
    PyTorchSession,
)
from gpu_algos import (
    gpu_resize_to_,
    gpu_resize_,
)


class _CACHE_STATE(IntEnum):
    INIT = 0x00
    WAIT_FRAMES = 0x01
    PROCESSING = 0x02
    EMPTYING = 0x03
    EMPTY = 0x04



class TemporalFrameCacheMirroring:

    def __init__(self, window_size: int = 7) -> None:
        self.window_size: int = window_size
        self.radius: int = window_size // 2
        self.frames: list[NnFrame] = []
        self.tensors: list[Tensor] = []
        self.tail: list[Tensor] = []   # padding miroir de fin
        self.pad_i: int = 0
        self.state: _CACHE_STATE = _CACHE_STATE.INIT


    def reset(self) -> None:
        self.frames.clear()
        self.tensors.clear()
        self.tail.clear()
        self.pad_i = 0
        self.state = _CACHE_STATE.INIT


    def emptying(self) -> bool:
        return bool(self.state == _CACHE_STATE.EMPTYING)


    def is_ready(self) -> bool:
        return bool(self.state in (_CACHE_STATE.PROCESSING, _CACHE_STATE.EMPTYING))


    def is_empty(self) -> bool:
        return bool(self.state in (_CACHE_STATE.EMPTY, _CACHE_STATE.INIT))


    def _enter_emptying(self) -> None:
        # tensors contient 7 éléments, la dernière frame du clip est en [-1]
        # padding miroir : f(N-2), f(N-3), f(N-4)
        self.tail = [self.tensors[-(k + 1)] for k in range(1, self.radius + 1)]
        self.pad_i = 0
        self.state = _CACHE_STATE.EMPTYING


    def append(self, frame: NnFrame | None) -> None:
        if frame is None:
            return

        if self.state == _CACHE_STATE.INIT:
            self.tensors = [frame.tensor]
            self.frames = [frame]
            self.state = _CACHE_STATE.WAIT_FRAMES

        elif self.state == _CACHE_STATE.WAIT_FRAMES:
            self.tensors.append(frame.tensor)
            self.frames.append(frame)
            # print(f"cache frame no: " + ", ".join(str(f.f_no) for f in self.frames))

            if len(self.frames) == self.radius + 1:
                # [f0 f1 f2 f3] -> [f3 f2 f1 f0 f1 f2 f3]
                prefix = self.tensors[self.radius:0:-1]
                self.tensors = prefix + self.tensors
                if frame.last:          # N == radius + 1
                    self._enter_emptying()
                else:
                    self.state = _CACHE_STATE.PROCESSING

        elif self.state == _CACHE_STATE.PROCESSING:
            if len(self.tensors) >= self.window_size:
                raise ValueError("too many tensors")
            self.tensors.append(frame.tensor)
            self.frames.append(frame)
            if frame.last:
                self._enter_emptying()

        else:
            raise ValueError(red("cache: append while not ready to receive"))


    def get_window(self) -> list[Tensor] | None:
        if self.state == _CACHE_STATE.PROCESSING:
            # print(f"generated a window with frame no: " + ", ".join(str(f.f_no) for f in self.frames))
            return self.tensors

        if self.state == _CACHE_STATE.EMPTYING:
            while len(self.tensors) < self.window_size:
                self.tensors.append(self.tail[self.pad_i])
                self.pad_i += 1
            return self.tensors

        return None


    def current_frame(self) -> NnFrame | None:
        if self.state in (_CACHE_STATE.PROCESSING, _CACHE_STATE.EMPTYING):
            frame: NnFrame = self.frames.pop(0)
            frame.tensor = None
            self.tensors.pop(0)
            if len(self.frames) == 0:
                self.state = _CACHE_STATE.EMPTY
            return frame
        return None




class CudaTemporalInferenceThread(BaseThread):
    def __init__(
        self,
        name: str | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(name=name, verbose=debug)


    def initialize(
        self,
        model: PyTorchModel | None,
        device: str = "cuda:0",
        dtype: Hdtype = 'fp32',
        prescale: tuple[int, int, int] | float | None = None,
    ) -> None:
        self.model = model
        self.cache = TemporalFrameCacheMirroring(window_size=model.arch.infer_type.inputs)
        self.cuda_session: PyTorchSession = nnlib.session(model=model)
        self.cuda_session.initialize(
            device=device,
            dtype=dtype,
            warmup=False,
        )
        self.infer_stream = torch.cuda.Stream(device)

        if isinstance(prescale, list | tuple):
            self.prescale = (prescale[0], prescale[1])
        else:
            self.prescale = prescale


    def run(self):
        self.perform_cuda_infer()
        while not self.in_queue.empty():
            self.in_queue.get_nowait()


    @torch.inference_mode()
    def perform_cuda_infer(self) -> None:
        verbose: bool = self.verbose
        if verbose:
            print(blue(f"[V][I][CUDA] Cuda TemporalInferenceThread"))
        in_queue: Queue = self.in_queue

        cache: TemporalFrameCacheMirroring = self.cache
        cuda_stream = self.infer_stream
        session_dtype: torch.dtype = self.cuda_session.dtype
        nn_module = self.cuda_session.module

        with torch.cuda.stream(cuda_stream):
            while not self._stop_event.is_set():
                if verbose:
                    dlogger.debug(blue("[V][I][CUDA] waiting"))

                frame = None
                if not cache.emptying():
                    # receive frame from consumer
                    input = in_queue.get(block=True)
                    if input is None or self._stop_event.is_set():
                        break
                    frame: NnFrame = input

                    wait_frame_ready(frame=frame, stream=cuda_stream)
                    if verbose:
                        verbose_prefix: str = f"[V][I][CUDA][{frame.f_no}]"

                    # Resize before inference
                    if self.prescale is not None:
                        if isinstance(self.prescale, list | tuple):
                            gpu_resize_to_(
                                frame=frame,
                                out_size=self.prescale,
                                interpolation_method="bicubic"
                            )
                        else:
                            gpu_resize_(
                                frame=frame,
                                scale_factor=self.prescale,
                                interpolation_method="bicubic"
                            )

                    # print(purple(f"received frame no.{frame.f_no}"), "last" if frame.last else "")
                    # Append frame to cache (resized)
                    in_tensor = frame.tensor
                    in_tensor = in_tensor.to(dtype=session_dtype)
                    frame.tensor = in_tensor.contiguous()
                    if verbose:
                        dlogger.debug(f"cache {frame.f_no}: {in_tensor.data_ptr()}")
                    cache.append(frame=frame)

                # elif verbose:
                #     dlogger.debug(blue(f"[V][I][CUDA] emptying"))

                # Do not process if cache is not ready
                if not cache.is_ready():
                    if verbose:
                        dlogger.debug(blue(f"[V][I][CUDA] ask for frame"))
                    self.producer.set_produce_flag()
                    continue

                # Cache is ready, perform inference
                window = cache.get_window()
                if verbose:
                    dlogger.debug(f"   window: {[t.data_ptr() for t in window]}")
                if window is None:
                    raise ValueError("Not enough frames in window, why?")

                # inference
                #-----------------------------------------------------------------------------
                if False:
                    # Debug
                    # out_tensor = (
                    #     window[0] * 0.25
                    #     + window[1] * 0.5
                    #     + window[2] * 0.75
                    #     + window[3]
                    #     + window[4] * 0.75
                    #     + window[5] * 0.5
                    #     + window[6] * 0.25
                    # ) / 4
                    out_tensor = window[3]
                else:
                    in_tensors = torch.cat(window, dim=0).unsqueeze(0)
                    out_tensor = nn_module(in_tensors)

                out_tensor = torch.clamp_(out_tensor, 0., 1.)
                out_frame = cache.current_frame()
                out_frame.tensor = out_tensor
                # print(f"encode {out_frame.f_no}")

                mark_frame_ready(frame=out_frame, stream=cuda_stream)
                if verbose:
                    print(blue(f"{verbose_prefix} send to encoder: {out_tensor.shape}, {out_tensor.dtype}, {out_tensor.data_ptr()}"))
                self.consumer.put_frame(out_frame)

                # Write (debug)
                # write_tensor(f"frame_{out_frame.f_no:02}.png", d_tensor=out_tensor)

                if cache.is_empty():
                    print(blue(f"[V][I][CUDA] cache is empty, break"))
                    break

        if verbose:
            print(blue(f"[V][I][CUDA] Ended"))

        # Send a poison pill ?
        self.consumer.put(None)
