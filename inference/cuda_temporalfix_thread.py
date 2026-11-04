from __future__ import annotations
from enum import IntEnum
from pprint import pprint
from queue import Queue
import time
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from hytils import blue, cyan, red
import torch.nn.functional as F

from denc import (
    BaseThread,
    NnFrame,
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
if TYPE_CHECKING:
    from denc import (
        DecoderThread,
        EncoderThread,
    )



class _CACHE_STATE(IntEnum):
    INIT = 0x00
    WAIT_FRAMES = 0x01
    PROCESSING = 0x02
    EMPTYING = 0x03
    EMPTY = 0x04



class TemporalFrameCache():

    def __init__(self, window_size: int = 5) -> None:
        self.window_size: int = window_size
        self.frames: list[NnFrame] = []
        self.tensors: list[Tensor] = []
        self.state: _CACHE_STATE = _CACHE_STATE.INIT


    def reset(self) -> None:
        self.frames.clear()
        self.tensors.clear()
        self.state = _CACHE_STATE.INIT


    def emptying(self) -> bool:
        return bool(self.state == _CACHE_STATE.EMPTYING)


    def is_ready(self) -> bool:
        return bool(
            self.state in (_CACHE_STATE.PROCESSING, _CACHE_STATE.EMPTYING)
        )


    def is_empty(self) -> bool:
        return bool(
            self.state in (_CACHE_STATE.EMPTY, _CACHE_STATE.INIT)
        )


    def append(self, frame: NnFrame | None) -> None:
        if frame is None:
            return

        if self.state == _CACHE_STATE.INIT:
            self.tensors = [frame.tensor] * (self.window_size // 2 + 1)
            self.frames = [frame]
            self.state = _CACHE_STATE.WAIT_FRAMES

        elif self.state == _CACHE_STATE.WAIT_FRAMES:
            self.tensors.append(frame.tensor)
            self.frames.append(frame)
            if len(self.tensors) >= self.window_size:
                self.state = _CACHE_STATE.PROCESSING

        elif self.state == _CACHE_STATE.PROCESSING:
            if len(self.frames) >= self.window_size:
                raise ValueError("Too many frames in cache")
            if len(self.tensors) >= self.window_size:
                raise ValueError("too many tensors")

            self.tensors.append(frame.tensor)
            self.frames.append(frame)
            if frame.last:
                self.state = _CACHE_STATE.EMPTYING

        else:
            raise ValueError(red("cache: append while not ready to receive"))


    def get_window(self) -> list[Tensor] | None:
        if self.state == _CACHE_STATE.PROCESSING:
            return self.tensors

        elif self.state == _CACHE_STATE.EMPTYING:
            to_add: int = (self.window_size - len(self.tensors))
            for _ in range(to_add):
                self.tensors.append(self.tensors[-1])
            return self.tensors

        return None


    def current_frame(self) -> NnFrame | None:
        if _CACHE_STATE.PROCESSING <= self.state < _CACHE_STATE.EMPTY:
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
        self.cache = TemporalFrameCache()
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
            print(cyan(f"[V][I][CUDA] Cuda InferenceThread"))
        in_queue: Queue = self.in_queue

        cache: TemporalFrameCache = self.cache
        cuda_stream = self.infer_stream
        session_dtype: torch.dtype = self.cuda_session.dtype
        nn_module = self.cuda_session.module

        with torch.cuda.stream(cuda_stream):
            while not self._stop_event.is_set():
                if verbose:
                    print(cyan("[V][I][CUDA] waiting"))

                frame = None
                if not cache.emptying():
                    # receive frame from consumer
                    input = in_queue.get(block=True)
                    if input is None or self._stop_event.is_set():
                        break
                    frame: NnFrame = input

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
                else:
                    print(red("emptying"))

                # Append fram to cache
                cache.append(frame=frame)

                # Do not process if cache is not ready
                if not cache.is_ready():
                    print("  ask for frame")
                    self.producer.set_produce_flag()
                    continue

                # Cache is ready, perform inference
                window = cache.get_window()
                if window is None:
                    raise ValueError("Not enough frames in window, why?")

                # inference
                #-----------------------------------------------------------------------------
                # debug
                # out_tensor = (
                #     window[0] * 0.5
                #     + window[1] * 0.75
                #     + window[2]
                #     + window[3] * 0.75
                #     + window[4] * 0.5
                # ) / 3.5
                out_tensor = nn_module(in_tensor)



                #-----------------------------------------------------------------------------
                out_tensor = torch.clamp_(out_tensor, 0., 1.)
                out_tensor = window[2]

                # get frame to output
                out_frame = cache.current_frame()
                out_frame.tensor = out_tensor
                gpu_resize_to_(out_frame, out_shape)

                time.sleep(0.0001)
                cuda_stream.synchronize()

                # Write (debug)
                # write_tensor(f"frame_{out_frame.f_no:02}.png", d_tensor=out_tensor)

                self.consumer.put_frame(out_frame)

                if cache.is_empty():
                    break

                if not cache.emptying():
                    self.producer.set_produce_flag()

        # Send a poison pill
        self.consumer.put(None)
