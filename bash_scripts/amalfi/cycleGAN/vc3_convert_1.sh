python -m cycleGAN_VC3.generate \
    --name source_voc_10_target_coraal_PRV_se0_ag2_f_03 \
    --save_dir /data1/cycleGAN_VC3/converted \
    --source_id 10 \
    --ckpt_path /data1/cycleGAN_VC3/source_voc_10_target_coraal_PRV_se0_ag2_f_03/ckpts/ \
    --load_epoch 6000
    # --normalized_dataset_A_path /data1/datasets/melspec_dataset/voc/10/voc_normalized.pickle \
    # --norm_stats_A_path /data1/datasets/melspec_dataset/voc/10/norm_stat_voc.npz \
    # --normalized_dataset_B_path /data1/datasets/melspec_dataset/coraal/PRV_se0_ag2_f_03/coraal_normalized.pickle \
    # --norm_stats_B_path /data1/datasets/melspec_dataset/coraal/PRV_se0_ag2_f_03/norm_stat_coraal.npz \
    # --num_frames_validation 320
