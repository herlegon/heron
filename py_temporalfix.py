from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from fractions import Fraction
import logging
import math
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
    PyTorchModel,
)
from hsys import (
    is_feature_supported
)




def main():
    dlogger.setLevel(logging.INFO)

    parser = ArgumentParser()
    add_io_arguments(parser)
    parser.add_argument(
        "-s",
        "--strength",
        type=BoundedFloat(0, 3),
        metavar="[0..3]",
        default=2,
        required=False,
        help="""Suppression strength of temporal inconsistencies in the range `0.0-3.0`. Higher means more aggressive.
Higher resolution tends to need higher strength. Too high may oversmooth small movements. Default: 2.
\n"""
    )
    parser.add_argument(
        "-mdir",
        "--model_dir",
        type=str,
        default="",
        required=False,
        help="""Model directory
\n"""
    )
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

    # Models
    #-------------------------------------------------------------------------------------
    if args.model_dir:
        model_dir: Path = Path(args.model_dir).resolve()
    else:
        # Specific for me
        model_dir = Path(__file__).resolve().parents[1] / "ml_models" / "temporalfix"
        if not model_dir.is_dir():
            model_dir: Path = Path(__file__).resolve().parent / "external" / "models"

    temporal_models: dict[int, str] = {
        0: "temporalfix_s0_v1.pth",
        1: "temporalfix_s1_v1.1.pth",
        2: "temporalfix_s2_v1.pth",
        3: "temporalfix_s3_v1.pth",
    }

    # TODO: download model if missing
    for m in temporal_models.values():
        model_fp = model_dir / m
        if not model_fp.is_file():
            dlogger.error(red(f"Missing model: {m}"))


    strength = round(float(args.strength), 2)
    strength_lower = math.floor(strength)
    strength_upper = math.ceil(strength)
    weighting = strength - strength_lower

    model: PyTorchModel = nnlib.open(model_path=temporal_models[strength_lower])
    if strength_lower != strength_upper:
        lower_model: PyTorchModel = model
        upper_model: PyTorchModel = nnlib.open(model_path=temporal_models[strength_upper])

        # lower_model.state_dict
        # upper_model.state_dict
        consolidated_state_dict = OrderedDict(
            (
                key, torch.lerp(value, upper_model.state_dict[key], weighting)
                if torch.is_floating_point(value) else value
            )
            for key, value in lower_model.state_dict.items()
        )

        # Let's patch the model state_dict
        model.state_dict = consolidated_state_dict
        # No need to keep the uppdermodel in memory
        del upper_model





if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if sys.platform != 'win32':
        print(red(f"Error: {sys.platform} is not a supported platform. But trying..."))
    main()


