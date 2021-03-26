python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc \
    --save_dir results/ \
    --num_epochs 6172 \
    --normalized_dataset_A_path vcc2018_training_preprocessed/VCC2SF3_normalized.pickle \
    --norm_stats_A_path vcc2018_training_preprocessed/VCC2SF3_norm_stat.npz \
    --normalized_dataset_B_path vcc2018_training_preprocessed/VCC2TF1_normalized.pickle \
    --norm_stats_B_path vcc2018_training_preprocessed/VCC2TF1_norm_stat.npz \
    --epochs_per_save 1 \
    --epochs_per_plot 10 \
    --batch_size 1 \
    --decay_after 10000 \
    --max_mask_len 25 \
    --gpu_ids 0
 