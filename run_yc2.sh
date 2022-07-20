DATA_PATH=YOUR_PATH

export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=30668 \
main_task_retrieval.py --do_train --num_thread_reader=8 \
--epochs=20 --batch_size=64 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/Activity_Videos \
--output_dir ckpts/ckpt_activity_retrieval_looseType \
--lr 1e-4 --coef_lr 5e-3 --max_words 128 --max_frames 64 --max_audio_frames 2 \
--audio_cluster 4 --batch_size_val 64 --gradient_accumulation_steps 1 \
--datatype youcook --feature_framerate 1  \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--yb_coff_dis 0 --yb_coff_loss 1 --yb_av 0 --yb_dual 0 --yb_factor_aduio 1 --yb_factor_audio_decay 0 --yb_audio_length 10 \
--pretrained_clip_name ViT-B/32  --wandb  1 --fixed_length 8 --model_name Rebuttal_yc2_B32_baseline_64 \
--audio_pt VGGSound_Audio_features_10s_aligned
