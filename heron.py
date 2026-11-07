from argparse import ArgumentParser
from copy import deepcopy
from fractions import Fraction
import logging
from pathlib import Path
import re
import sys
import signal
from warnings import warn

from denc.vstream import FFmpegResizer
from hytils import blue, green, lightcyan, lightgreen, purple, red, yellow
from inference.cuda_infer_thread import CudaInferThread
from inference.run_threads import run_threads
from inference.trt_infer_thread import TrtInferThread
try:
    import winsound
except ImportError:
    pass
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denc
from denc import (
    MediaStream,
    VideoStreamFormat,
    PipeFormat,
    dlogger,
    DecoderThread,
    EncoderThread,
    ProgressThread,
    Device,
    add_io_arguments,
    add_model_arguments,
    add_debug_arguments,
    add_sar_arguments,
    add_resize_arguments,
    add_inference_arguments,
    generate_output_filepath,
    BoundedFloat,
)
from nnarchs import (
    nnlib,
    NnModel,
    Hdtype,
    HdtypeToTorch,
    NnFrameworkType,
)
from hsys import (
    is_feature_supported,
    VramMonitor,
)




def main():
    dlogger.setLevel(logging.INFO)

    parser = ArgumentParser()
    add_io_arguments(parser)
    add_model_arguments(parser)
    add_sar_arguments(parser)
    add_resize_arguments(parser)
    add_inference_arguments(parser)

    add_debug_arguments(parser)
    args = parser.parse_args()
    debug: bool = args.debug
    if debug:
        dlogger.setLevel(logging.DEBUG)


    # Input file
    #-------------------------------------------------------------------------------------
    in_fp: Path = Path(args.input).resolve()
    in_media: MediaStream = denc.open(filepath=in_fp)
    dlogger.info(f"{lightcyan(f"Input file:")}  {in_fp}")
    dlogger.debug(in_media.video)

    # Output file
    #-------------------------------------------------------------------------------------
    output_fp = generate_output_filepath(args)

    # Output media
    #-------------------------------------------------------------------------------------
    out_media = in_media.to_output(filepath=output_fp)
    dlogger.debug(out_media.video)

    # Model
    #-------------------------------------------------------------------------------------
    if not args.model:
        raise ValueError(red(f"missing model"))

    # Chain
    #-------------------------------------------------------------------------------------

    # Resize with FFmpeg before running the algo
    #   adjust the width (most common use case)
    in_media.video.resize.enabled = True
    in_media.video.resize.use_sar = True
    in_media.video.resize.side = 'width'

    # Verify that the hardware supports
    device = "cuda:0"
    if args.cpu:
        device = 'cpu'
        in_media.video.device = Device(name=device)

    else:
        if not is_feature_supported(feature='cuda'):
            dlogger.warning(ValueError(yellow(f"No Nvidia GPU found, using CPU")))
            device = 'cpu'
            in_media.video.device = Device(name=device)

        else:
            # use cuda device 0 by default
            in_media.video.device = Device(name=device)
            out_media.video.device = Device(name=device)

    # Resize
    #-------------------------------------------------------------------------------------
    step_no: int = 1

    # Update output video size if needed when SAR is not 1:1
    do_resize: bool = False
    h, w, c = in_media.video.shape
    if args.fsar and args.fsar_h:
        sys.exit(red("[E] fsar and fsar_h cannot be used together"))

    # Let the user choose how to resize when sar != 1:1
    do_resize_with_fsar: bool = args.fsar or args.fsar_h
    if do_resize_with_fsar:
        size: tuple[int, int] = ()
        # resize width
        if result := re.search(re.compile(r"^(\d+)\/(\d+)$"), args.fsar):
            size = (int(w * int(result.group(2)) / int(result.group(1)) + 0.5), h)
        # resize height
        if result := re.search(re.compile(r"^(\d+)\/(\d+)$"), args.fsar_h):
            size = (w, int(h * int(result.group(1)) / int(result.group(2)) + 0.5))

        in_media.video.resize = FFmpegResizer(
            enabled=True,
            use_sar=True,
            side='height' if args.fsar_h else 'width',
            size=size
        )

        print(f" {step_no}. resize to {'x'.join(size)}")
        step_no += 1

    # Otherwise resize with sar (Default: resize height)
    if not do_resize_with_fsar and in_media.video.sar != Fraction(1, 1):
        do_resize = True
        size = (w, int(h * in_media.video.sar.numerator / in_media.video.sar.denominator))

        print(f" {step_no}. resize to {'x'.join(size)}")
        step_no += 1

    if do_resize:
        warn(yellow(f"[W] TODO: validate resize: {in_media.video.shape} -> {size}"))

    # Resize before inference
    #-------------------------------------------------------------------------
    in_sf: VideoStreamFormat = in_media.video.stream_format()
    dlogger.debug(green(f"Decoder stream format: ") + f"{in_sf}")
    n1_sf = deepcopy(in_sf)

    if args.resize != 1:
        do_resize = True
        resize_factor = args.resize
        h, w, c = n1_sf.shape
        n1_sf.shape = (int(resize_factor * h + 0.5), int(resize_factor * w + 0.5), c)

    elif args.resize_to:
        do_resize = True
        w, h = args.resize_to.split("x")
        n1_sf.shape = (int(h), int(w), n1_sf.shape[-1])

    pre_resize_shape: list | None = None
    if do_resize:
        pre_resize_shape = n1_sf.shape
        print(f" {step_no}. resize to {n1_sf.shape[1]}x{n1_sf.shape[0]}")
        step_no += 1

    dlogger.debug(blue(f"Prescale stream format: ") + f"{n1_sf}")

    # Open model
    #-------------------------------------------------------------------------
    vram_monitor = VramMonitor(device=0)
    vram_monitor.start()

    n2_sf = deepcopy(n1_sf)
    model_fp = Path(args.model).resolve()
    model: NnModel = nnlib.open(model_fp, device)
    dlogger.info(lightcyan(f"\nModel: ") + f"{model_fp}")
    dlogger.debug(model)
    dlogger.info(blue(f"  framework: ") + model.framework.type.value)
    dlogger.info(blue(f"  arch: ") + f"{model.arch_name}")
    dlogger.info(blue(f"  scale: ") + f"{model.scale}")

    infer_dtype: Hdtype = 'fp32'
    if args.fp32:
        infer_dtype = 'fp32'
    elif args.fp16:
        infer_dtype = 'fp16'
    elif args.bf16:
        infer_dtype = 'bf16'

    h, w, c = n2_sf.shape
    n2_sf.shape = (model.scale * h, model.scale * w, c)
    n2_sf.dtype = HdtypeToTorch[infer_dtype]
    dlogger.debug(blue(f"Inference output stream format: ") + f"{n2_sf}")
    print()

    # Decoder
    #-------------------------------------------------------------------------
    decoder_thread = DecoderThread(
        media_stream=in_media,
        verbose=debug
    )

    # Encoder
    #-------------------------------------------------------------------------
    out_media: MediaStream = in_media.to_output(filepath=output_fp)
    out_sf = deepcopy(n2_sf)
    out_media.video.input_stream_format = out_sf
    dlogger.debug(purple(f"Encoder input stream format: ") + f"{out_sf}")
    dlogger.debug(f"input->FFmpeg pipe: {out_media.video.pipe_format}")
    encoder_thread = EncoderThread(
        media_stream=out_media,
        verbose=debug
    )

    # Inference threads: currently a single one
    #-------------------------------------------------------------------------
    infer_threads: list[TrtInferThread | CudaInferThread] = []

    # Tensor inference thread: SISR
    if model.framework.type == NnFrameworkType.TENSORRT:
        infer_threads.append(
            TrtInferThread(
                name=model.arch.name,
                debug=debug
            )
        )
        infer_threads[-1].initialize(
            model=model,
            device="cuda:0",
            dtype=infer_dtype,
            prescale=pre_resize_shape
        )

    # Cuda inference thread: SISR
    if model.framework.type == NnFrameworkType.PYTORCH:
        infer_threads.append(CudaInferThread(
                name=model.arch.name,
                debug=debug
            )
        )
        infer_threads[-1].initialize(
            model=model,
            device="cuda:0",
            dtype=infer_dtype,
            prescale=None
        )

    # Set producers and consumers
    #-------------------------------------------------------------------------
    # Currently easier and safe to connect the encoder to the decoder
    # Only a single inference thread
    decoder_thread.set_consumer(infer_threads[0])
    infer_threads[0].set_consumer(encoder_thread)
    encoder_thread.set_producer(decoder_thread)

    # Progress bar
    #-------------------------------------------------------------------------
    total_frames: int = in_media.video.frame_count
    progress_thread: ProgressThread = ProgressThread(total=total_frames)
    encoder_thread.set_progress_thread(progress_thread)

    # Main loop
    #-------------------------------------------------------------------------
    stats = {}

    total_frames: int = out_media.video.frame_count
    elapsed = run_threads(
        dec_thread=decoder_thread,
        enc_thread=encoder_thread,
        infer_threads=infer_threads,
        progress_thread=progress_thread,
        total_frames=total_frames,
        vram_stats=stats,
        debug=debug,
        vram_monitor=vram_monitor
    )

    print(f"elapsed: {elapsed:.02f}s ({total_frames/elapsed:.02f}fps)")
    print(f"VRAM mean: {stats['mean_mb']:.0f} MB | peak: {stats['peak_mb']:.0f} MB")

    if not args.cpu:
        torch.cuda.empty_cache()

    if sys.platform == "win32":
        winsound.Beep(frequency=440, duration=200)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
