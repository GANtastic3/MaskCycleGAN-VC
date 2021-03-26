python -W ignore::UserWarning -m cycleGAN_VC3.train \
    --name source_voc_18_target_coraal_PRV_se0_ag3_m_03 \
    --save_dir /data1/cycleGAN_VC3 \
    --num_epochs 6172 \
    --normalized_dataset_A_path /data1/datasets/melspec_dataset/voc/18/voc_normalized.pickle \
    --norm_stats_A_path /data1/datasets/melspec_dataset/voc/18/norm_stat_voc.npz \
    --normalized_dataset_B_path /data1/datasets/melspec_dataset/coraal/PRV_se0_ag3_m_03/coraal_normalized.pickle \
    --norm_stats_B_path /data1/datasets/melspec_dataset/coraal/PRV_se0_ag3_m_03/norm_stat_coraal.npz \
    --epochs_per_save 250 \
    --epochs_per_plot 25 \
    --batch_size 1 \
    --decay_after 10000 \
    --max_mask_len 25
    # --num_frames_validation 320
