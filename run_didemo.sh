# bash mykill.sh
DATA_PATH=YOUR_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=30678 \
main_task_retrieval.py --do_train --num_thread_reader=6 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/raw_video \
--output_dir ckpts/ckpt_didemo_retrieval_looseType \
--lr 1e-4 --coef_lr 5e-3 --yb_factor_aduio 0.01 --yb_factor_audio_decay 0 --max_words 64 --max_frames 32 --max_audio_frames 32 \
--batch_size_val 16 \
--datatype didemo --feature_framerate 1  \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--yb_dual 0 --yb_av 1 --wandb 0 --model_name ECLIPSE_didemo_f32_B32 \
--pretrained_clip_name ViT-B/32
