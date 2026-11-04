from __future__ import annotations
from queue import Queue
import time
from typing import TYPE_CHECKING
import torch
from torch import Tensor
from hytils import blue, cyan

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



class CudaInferThread(BaseThread):
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
    ):
        """Isolate for fps measurement"""
        self.cuda_session: PyTorchSession = nnlib.session(model)
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
    def perform_cuda_infer(self):
        verbose: bool = self.verbose
        if verbose:
            print(cyan(f"[V][I][CUDA] Cuda InferenceThread"))
        in_queue: Queue = self.in_queue

        # d_thread: DecoderThread = self.producer

        cuda_stream = self.infer_stream
        session_dtype: torch.dtype = self.cuda_session.dtype
        nn_module = self.cuda_session.module
        # scale = session.model.scale

        with torch.cuda.stream(cuda_stream):
            while not self._stop_event.is_set():
                if verbose:
                    print(cyan("[V][I][CUDA] waiting"))
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

                # Input tensor
                in_tensor = frame.tensor
                # n, c, in_h, in_w = in_tensor.shape
                in_tensor = in_tensor.to(dtype=session_dtype)
                in_tensor = in_tensor.contiguous()
                # in_tensor = in_tensor.to(dtype=...)
                if verbose:
                    print(blue(f"{verbose_prefix} in tensor dtype:{in_tensor.dtype}, {in_tensor.data_ptr()}"))

                # Perform simple inference
                frame.tensor = nn_module(in_tensor)
                # frame = gpu_resize_(frame, scale_factor=2.)
                frame.tensor = torch.clamp_(frame.tensor, 0., 1.)

                time.sleep(0.0001)
                cuda_stream.synchronize()

                self.consumer.put_frame(frame)

                # faster but race condition?
                # if d_thread is not None:
                #     d_thread.set_produce_flag()

        if verbose:
            print(cyan(f"[V][I][CUDA] Ended"))
