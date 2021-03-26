python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc \
    --save_dir results/ \
    --num_epochs 6172 \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TF1 \
    --epochs_per_save 1 \
    --epochs_per_plot 10 \
    --batch_size 1 \
    --decay_after 10000 \
    --max_mask_len 25 \
    --gpu_ids 0
 