python -W ignore::UserWarning -m cycleGAN_VC3.train \
    --name source_voc_24_target_coraal_ATL_se0_ag1_f_01 \
    --save_dir /data1/cycleGAN_VC3 \
    --num_epochs 6172 \
    --normalized_dataset_A_path /data1/datasets/melspec_dataset/voc/24/voc_normalized.pickle \
    --norm_stats_A_path /data1/datasets/melspec_dataset/voc/24/norm_stat_voc.npz \
    --normalized_dataset_B_path /data1/datasets/melspec_dataset/coraal/DCB_se1_ag2_f_01/coraal_normalized.pickle \
    --norm_stats_B_path /data1/datasets/melspec_dataset/coraal/DCB_se1_ag2_f_01/norm_stat_coraal.npz \
    --epochs_per_save 250 \
    --epochs_per_plot 25 \
    --batch_size 1 \
    --decay_after 10000 \
    --max_mask_len 25 \
    --gpu_ids 1
    # --num_frames_validation 320
