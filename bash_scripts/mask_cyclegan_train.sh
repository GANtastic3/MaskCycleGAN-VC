# Sample training script to convert between VCC2SF3 and VCC2TF1
# Continues training from epoch 500

python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TF1 \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TF1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --batch_size 1 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0 \
