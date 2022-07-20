DATA_PATH=/playpen-iop/yblin/didemo_data
export CUDA_VISIBLE_DEVICES=4,5,6,7



python -m torch.distributed.launch --nproc_per_node=4 --master_port=9864 \
main_task_retrieval.py --do_train --num_thread_reader=8 \
--epochs=20 --batch_size=64 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/raw_video \
--output_dir ckpts/ckpt_didemo_retrieval_looseType \
--lr 1e-4 --coef_lr 1e-3 --yb_factor_aduio 0.5 --yb_factor_audio_decay 0.6 --max_words 64 --max_frames 8 --max_audio_frames 8 --batch_size_val 16 \
--datatype CUSTOM_DATASET --feature_framerate 1  \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--yb_dual 1 --yb_av 1 --wandb 1 --model_name Rebuttal_qvh_ours-8f \
--pretrained_clip_name ViT-B/32