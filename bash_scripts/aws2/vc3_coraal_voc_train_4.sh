python -W ignore::UserWarning -m cycleGAN_VC3.train \
    --name source_voc_26_target_coraal_PRV_se0_ag2_m_01 \
    --save_dir /home/ubuntu/results/cycleGAN_VC3 \
    --num_epochs 6172 \
    --normalized_dataset_A_path /home/ubuntu/data/melspec_dataset/voc/26/voc_normalized.pickle \
    --norm_stats_A_path /home/ubuntu/data/melspec_dataset/voc/26/norm_stat_voc.npz \
    --normalized_dataset_B_path /home/ubuntu/data/melspec_dataset/coraal/PRV_se0_ag2_m_01/coraal_normalized.pickle \
    --norm_stats_B_path /home/ubuntu/data/melspec_dataset/coraal/PRV_se0_ag2_m_01/norm_stat_coraal.npz \
    --epochs_per_save 250 \
    --epochs_per_plot 100 \
    --batch_size 1 \
    --decay_after 10000 \
    --max_mask_len 25 \
    --gpu_ids 0
    # --num_frames_validation 320
