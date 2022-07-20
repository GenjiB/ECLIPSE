DATA_PATH=/playpen-iop/yblin/qvhighlight
DATA_PATH=/playpen-storage/yblin/v1-2
export CUDA_VISIBLE_DEVICES=0

# bash mykill.sh

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=7455 \
yb_feature_ext.py --do_train --num_thread_reader=32 \
--epochs=1 --batch_size=1 --batch_size_val 1 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/Activity_Videos \
--output_dir ckpts/ckpt_activity_retrieval_looseType \
--lr 1e-4 --coef_lr 1e-3 --max_words 64 --max_frames 64 --max_audio_frames 32 \
--audio_cluster 16 --gradient_accumulation_steps 1 \
--datatype activity --feature_framerate 1  \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP --yb_audio_length 10 \
--pretrained_clip_name ViT-B/32  --wandb 0 --fixed_length 8 --model_name DistKL_avblock_ACLS+baseline_8 \
--yb_av 1 --yb_dual 1 --yb_start_layer 6 \
--audio_pt VGGSound_Audio_features_10s_aligned 



# DATA_PATH=/playpen-iop/yblin/v1-2
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=7445 \
# yb_feature_ext.py --do_train --num_thread_reader=0 \
# --epochs=1 --batch_size=1 --n_display=50 \
# --data_path ${DATA_PATH} \
# --features_path ${DATA_PATH}/Activity_Videos \
# --output_dir ckpts/ckpt_activity_retrieval_looseType \
# --lr 1e-4 --coef_lr 1e-3 --max_words 64 --max_frames 2 --max_audio_frames 2 \
# --audio_cluster 16 --batch_size_val 1 --gradient_accumulation_steps 1 \
# --datatype youcook --feature_framerate 1  \
# --freeze_layer_num 0  --slice_framepos 2 \
# --loose_type --linear_patch 2d --sim_header meanP --yb_audio_length 10 \
# --pretrained_clip_name ViT-B/32  --wandb 0 --fixed_length 8 --model_name DistKL_avblock_ACLS+baseline_8 \
# --audio_pt VGGSound_Audio_features_10s_aligned 