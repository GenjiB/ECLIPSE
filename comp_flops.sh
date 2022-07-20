DATA_PATH=/playpen-iop/yblin/v1-2
export CUDA_VISIBLE_DEVICES=4
python3 -m torch.distributed.launch --nproc_per_node=1  --master_port=30678 \
compute_cost.py --do_train --num_thread_reader=0 \
--epochs=20 --batch_size=64 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/Activity_Videos \
--output_dir ckpts/ckpt_activity_retrieval_looseType \
--lr 1e-4 --coef_lr 1e-3 --max_words 64 --max_frames 96 --max_audio_frames 96 \
--audio_cluster 4 --batch_size_val 1 --gradient_accumulation_steps 1 \
--datatype activity --feature_framerate 1  \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--yb_coff_dis 0 --yb_coff_loss 1 --yb_factor_1 0.001 --yb_audio_length 10 \
--pretrained_clip_name ViT-B/32  --wandb 0 --fixed_length 8 --model_name VGGSound_10s_co-Res_-5block_baseline_16 \
--audio_pt VGGSound_Audio_features_10s_aligned
