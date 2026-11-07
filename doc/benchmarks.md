# Commands:

## CUDA
cls; python .\heron.py --input A:\video-patterns\smpte_640x480_h265_yuv420p10_25fps_bt709_limited.mkv --model  A:\ml_models\2x-LD-Anime-Compact.pth --fp16 --cuda


## TensorRT
cls; python .\heron.py --input A:\video-patterns\smpte_640x480_h265_yuv420p10_25fps_bt709_limited.mkv --model A:\ml_models\2x-LD-Anime-Compact_cc8.9_op23_fp16_static_640x480_11.4.0.106.trtzip --fp16
