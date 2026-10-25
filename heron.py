from argparse import ArgumentParser
import logging
from pathlib import Path
import sys
import signal
from hytils import lightcyan, red, yellow
try:
    import winsound
except ImportError:
    pass
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import denc
from denc import (
    MediaStream,
    dlogger,
    DecoderThread,
    EncoderThread,
    ProgressThread,
    Device,
    add_io_arguments,
    add_debug_arguments,
)

from hsys import (
    is_feature_supported
)




def main():
    dlogger.setLevel(logging.INFO)

    parser = ArgumentParser()
    add_io_arguments(parser)

    add_debug_arguments(parser)
    args = parser.parse_args()
    debug: bool = args.debug
    if debug:
        dlogger.setLevel(logging.DEBUG)


    # Input file
    in_fp: Path = Path(args.input).resolve()
    in_media: MediaStream = denc.open(in_fp)
    dlogger.info(f"{lightcyan(f"Input file:")}  {in_fp}")
    dlogger.debug(in_media.video)

    # Output file
    #   priority: `output` -> `out_dir` -> none
    output: str = args.output
    out_dir: str = args.out_dir
    append_params: bool = args.append_params
    if output:
        output_fp: Path = Path(output).resolve()
    elif out_dir:
        output_fp: Path = Path(out_dir).resolve() / in_fp.name
    else:
        output_fp: Path = in_fp.parent / f"{in_fp.stem}_out{in_fp.suffix}"
        append_params = False

    if append_params:
        output_fp: Path = output_fp.parent / f"{output_fp.stem}_out{output_fp.suffix}"

    if output_fp == in_fp:
        dlogger.error(red(f"Error: Output file must not be the input file"))
        sys.exit()
    dlogger.info(f"{lightcyan(f"Output file:")} {output_fp}")

    out_media = in_media.to_output(filepath=output_fp)
    dlogger.debug(out_media.video)


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

    # Decoder
    #-------------------------------------------------------------------------
    decoder_thread = DecoderThread(
        media_stream=in_media,
        verbose=debug
    )

    # Encoder
    #-------------------------------------------------------------------------
    encoder_thread = EncoderThread(
        media_stream=out_media,
        verbose=debug
    )



    # Set producers and consumers
    #-------------------------------------------------------------------------
    decoder_thread.set_consumer(infer_thread)
    infer_thread.set_consumer(encoder_thread)
    encoder_thread.set_producer(decoder_thread)

    # Progress bar
    #-------------------------------------------------------------------------
    total_frames: int = in_media.video.frame_count
    progress_thread: ProgressThread = ProgressThread(total=total_frames)
    encoder_thread.set_progress_thread(progress_thread)

    # Main loop
    #-------------------------------------------------------------------------
    total_frames: int = out_media.video.frame_count
    elapsed = run_threads(
        decoder_thread=decoder_thread,
        encoder_thread=encoder_thread,
        i_threads=(infer_thread,),
        progress_thread=progress_thread,
        total_frames=total_frames,
        debug=debug
    )

    print(f"elapsed: {elapsed:.02f}s ({total_frames/elapsed:.02f}fps)")

    if not args.cpu:
        torch.cuda.empty_cache()

    if sys.platform == "win32":
        winsound.Beep(frequency=440, duration=200)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
