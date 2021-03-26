python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TF1 \
    --save_dir /data1/cycleGAN_VC3 \
    --num_epochs 6172 \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TF1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --batch_size 1 \
    --decay_after 10000 \
    --max_mask_len 25 \
    --gpu_ids 0
 