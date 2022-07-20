DATA_PATH=/playpen-iop/yblin/v1-2
export CUDA_VISIBLE_DEVICES=4
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=5277 \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=1 --batch_size=64 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/Activity_Videos \
--output_dir ckpts/ckpt_activity_retrieval_looseType \
--lr 1e-4 --coef_lr 1e-3 --max_words 64 --max_frames 2 --max_audio_frames 2 \
--audio_cluster 16 --batch_size_val 1 --gradient_accumulation_steps 1 \
--datatype qvhighlight --feature_framerate 1  \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP --yb_audio_length 1 --yb_dual 0 --yb_av 0 \
--pretrained_clip_name ViT-B/32  --wandb 0 --fixed_length 8 --model_name DistKL_avblock_ACLS+baseline_8 \
--audio_pt VGGSound_Audio_features_20s_aligned 
# --audio_pt Audio_features_all_10s_aligned_stride_16

