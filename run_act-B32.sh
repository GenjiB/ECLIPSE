# bash mykill.sh
# train on 4 gpus
DATA_PATH=YOUR_DATA_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3


python3 -m torch.distributed.launch --nproc_per_node=4  --master_port=7414 \
main_task_retrieval.py --do_train --num_thread_reader=6 \
--epochs=40 --batch_size=64 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/Activity_Videos \
--output_dir ckpts/ckpt_activity_retrieval_looseType \
--lr 1e-4 --coef_lr 1e-3 --max_words 128 --max_frames 32 --max_audio_frames 32 \
--audio_cluster 4 --batch_size_val 8 --gradient_accumulation_steps 1 \
--datatype activity --feature_framerate 1  \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--yb_av 1 --yb_dual 1 --yb_factor_aduio 0.5 --yb_time_cross_audio 0 --yb_factor_audio_decay 0.2 --yb_reverse_norm 1 \
--yb_audio_length 10 \
--pretrained_clip_name ViT-B/32  --wandb 1 --fixed_length 8 --model_name  ECLIPSE_act_f32_B32 \
--audio_pt VGGSound_Audio_features_10s_aligned


