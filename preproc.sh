# python3 preprocess/compress_video.py --input_root /playpen-iop/yblin/yk2/raw_videos_all/training --output_root  /playpen-iop/yblin/yk2/raw_videos_all/low_scale_train
# python3 preprocess/compress_video.py --input_root /playpen-iop/yblin/yk2/raw_videos_all/validation --output_root  /playpen-iop/yblin/yk2/raw_videos_all/low_scale_val



python3 preprocess/compress_video.py --input_root /playpen-iop/yblin/yk2/raw_videos_all/low_all_train --output_root  /playpen-iop/yblin/yk2/audio_raw_train
python3 preprocess/compress_video.py --input_root /playpen-iop/yblin/yk2/raw_videos_all/low_all_val --output_root  /playpen-iop/yblin/yk2/audio_raw_val

# python3 preprocess/compress_video.py --input_root /playpen-iop/yblin/yk2/raw_videos_all/low_scale_train --output_root /playpen-iop/yblin/yk2/raw_videos_all/low_all_train
# python3 preprocess/compress_video.py --input_root /playpen-iop/yblin/yk2/raw_videos_all/low_scale_val --output_root  /playpen-iop/yblin/yk2/raw_videos_all/low_all_val


# python3 preprocess/compress_video.py --input_root /playpen-iop/yblin/yk2/raw_videos_all/low_all_train --output_root /playpen-iop/yblin/yk2/raw_videos_all/videos_frame_train
# python3 preprocess/compress_video.py --input_root /playpen-iop/yblin/yk2/raw_videos_all/low_all_val --output_root  /playpen-iop/yblin/yk2/raw_videos_all/videos_frame_val