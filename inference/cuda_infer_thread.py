from __future__ import annotations
from queue import Queue
import torch
from hytils import blue, cyan, red

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
        try:
            self.perform_cuda_infer()
            while not self.in_queue.empty():
                self.in_queue.get_nowait()

        except Exception as e:
            dlogger.error(red(f"[CUDA] {str(e)}"))


    @torch.inference_mode()
    def perform_cuda_infer(self):
        verbose: bool = self.verbose
        if verbose:
            print(blue(f"[V][I][CUDA] Cuda InferenceThread"))
        in_queue: Queue = self.in_queue

        cuda_stream = self.infer_stream
        session_dtype: torch.dtype = self.cuda_session.dtype
        nn_module = self.cuda_session.module

        with torch.cuda.stream(cuda_stream):
            while not self._stop_event.is_set():
                if verbose:
                    print(blue("[V][I][CUDA] waiting"))

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

                # Input tensor
                in_tensor = frame.tensor
                in_tensor = in_tensor.to(dtype=session_dtype)
                in_tensor = in_tensor.contiguous()
                if verbose:
                    print(blue(f"{verbose_prefix} in tensor:{in_tensor.shape}, {in_tensor.dtype}, {in_tensor.data_ptr()}"))

                # Perform simple inference
                frame.tensor = nn_module(in_tensor)
                frame.tensor = torch.clamp_(frame.tensor, 0., 1.)

                mark_frame_ready(frame=frame, stream=cuda_stream)
                self.consumer.put_frame(frame)


        if verbose:
            print(blue(f"[V][I][CUDA] Ended"))
