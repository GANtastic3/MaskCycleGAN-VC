# Librispeech + finetuned on CoRAAL + voice converted 7 voc speakers

python -W ignore::UserWarning -m asr.main \
    --name asr_coraal_converted \
    --data_dir ~/data \
    --save_dir ~/results \
    --coraal \
    --converted \
    --num_epochs 100 \
    --batch_size 10 \
    --gpu_ids 0 \
    --num_workers 1 \
    --n_feats 80 \
    --epochs_per_save 5 \
    --pretrained_ckpt_path ~/results/librispeech/ckpts/best.pth.tar \
