from __future__ import annotations
from queue import Queue
import time
from typing import TYPE_CHECKING
import torch
from torch import Tensor

from hytils import blue, cyan
try:
    import tensorrt as trt
except:
    try:
        import tensorrt_rtx as trt
    except:
        pass
import torch

from denc import (
    BaseThread,
    NnFrame,
    wait_frame_ready,
    mark_frame_ready,
)
from nnarchs import (
    nnlib,
    Hdtype,
    TensorRtSession,
    TrtModel,
)
from gpu_algos import (
    gpu_resize_to_,
    gpu_resize_,
)


class TrtInferThread(BaseThread):
    def __init__(
        self,
        name: str | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(name=name, verbose=debug)


    def initialize(
        self,
        model: TrtModel,
        device: str = "cuda:0",
        dtype: Hdtype = 'fp16',
        prescale: list[int] | float | None = None,
    ):
        """Isolate for fps measurement"""

        self.trt_session: TensorRtSession = nnlib.session(model)
        self.trt_session.infer_stream = torch.cuda.Stream(device)
        self.trt_session.initialize(
            device=device,
            dtype=dtype,
            warmup=False,
        )

        if isinstance(prescale, list | tuple):
            self.prescale = (prescale[0], prescale[1])
        else:
            self.prescale = prescale


    def run(self):
        self.perform_trt_inference()
        while not self.in_queue.empty():
            self.in_queue.get_nowait()


    @torch.inference_mode()
    def perform_trt_inference(self, verbose: bool = False):
        verbose = self.verbose

        if verbose:
            print(cyan(f"[V][I][TRT] TensorRT InferenceThread"))
        in_queue: Queue = self.in_queue

        session = self.trt_session
        context, engine = session.context, session.engine
        session_dtype: torch.dtype = session.dtype
        cuda_stream = session.infer_stream

        scale = session.model.scale

        with torch.cuda.stream(cuda_stream):
            while not self._stop_event.is_set():
                if verbose:
                    print(cyan("[V][I][TRT] waiting"))
                input = in_queue.get(block=True)
                if input is None or self._stop_event.is_set():
                    break
                frame: NnFrame = input
                wait_frame_ready(frame=frame, cuda_stream=cuda_stream)

                if verbose:
                    verbose_prefix: str = f"[V][I][TRT][{frame.f_no}]"

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
                    time.sleep(0.0001)
                    cuda_stream.synchronize()

                # Input tensor
                in_tensor = frame.tensor
                n, c, in_h, in_w = in_tensor.shape
                in_tensor = in_tensor.to(dtype=session_dtype)
                in_tensor = in_tensor.contiguous()
                # in_tensor = torch.ravel(in_tensor)
                if verbose:
                    print(blue(
                        f"{verbose_prefix} in tensor: {in_tensor.shape}, {in_tensor.dtype}, {in_tensor.data_ptr()}"
                    ))

                # Prepare output tensor in same device
                out_tensor_shape = (n, c, in_h * scale, in_w * scale)
                out_tensor: torch.Tensor = torch.empty(
                    out_tensor_shape,
                    dtype=session_dtype,
                    device=in_tensor.device
                )

                # Perform simple inference
                bindings = [in_tensor.data_ptr(), out_tensor.data_ptr()]
                if verbose:
                    print(blue(f"{verbose_prefix} bindings: {bindings}, session dtype: {session_dtype}"))

                for i in range(engine.num_io_tensors):
                    tensor_name = engine.get_tensor_name(i)
                    context.set_tensor_address(tensor_name, bindings[i])
                    if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                        context.set_input_shape(tensor_name, in_tensor.shape)

                context.execute_async_v3(stream_handle=cuda_stream.cuda_stream)
                frame.tensor = torch.clamp(out_tensor, 0., 1.)

                mark_frame_ready(frame=frame, stream=cuda_stream)
                self.consumer.put_frame(frame)

        if verbose:
            print(cyan(f"[V][I][TRT] Ended"))
