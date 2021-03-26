python -W ignore::UserWarning -m cycleGAN_VC3.train \
    --name debug_mask_cyclegan_vc3 \
    --save_dir /data1/cycleGAN_VC3 \
    --num_epochs 6172 \
    --normalized_dataset_A_path /data1/datasets/vcc_2018_melspec/dataset_A_normalized.pickle \
    --norm_stats_A_path /data1/datasets/vcc_2018_melspec/norm_stat_A.npz \
    --normalized_dataset_B_path /data1/datasets/vcc_2018_melspec/dataset_B_normalized.pickle \
    --norm_stats_B_path /data1/datasets/vcc_2018_melspec/norm_stat_B.npz \
    --epochs_per_save 1 \
    --epochs_per_plot 10 \
    --batch_size 1 \
    --decay_after 10000 \
    --max_mask_len 25 \
    --continue_train \
    --load_epoch 1
    # --num_frames_validation 320