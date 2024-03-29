# bash mykill.sh
DATA_PATH=YOUR_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3




python -m torch.distributed.launch --nproc_per_node=4 --master_port=5278 \
main_task_retrieval.py --do_train --num_thread_reader=8 \
--epochs=20 --batch_size=64 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/raw_video \
--output_dir ckpts/ckpt_didemo_retrieval_looseType \
--lr 1e-4 --coef_lr 1e-3 --yb_factor_aduio 0.5 --yb_factor_audio_decay 0.6 --max_words 128 --max_frames 32 \
--max_audio_frames 32 --batch_size_val 16 \
--datatype qvhighlight --feature_framerate 1  \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--yb_dual 0 --yb_av 1 --wandb 0 --model_name ECLIPSE_qvh_f32_B32 \
--pretrained_clip_name ViT-B/32 --audio_pt VGGSound_Audio_features_10s_aligned 
