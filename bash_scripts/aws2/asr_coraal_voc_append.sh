# Librispeech + finetuned on CoRAAL + voice non-converted 29 voc speakers + voice converted 7 voc speakers

python -W ignore::UserWarning -m asr.main \
    --name asr_coraal_voc_append \
    --data_dir ~/data \
    --save_dir ~/results \
    --coraal \
    --voc \
    --append \
    --num_epochs 100 \
    --batch_size 36 \
    --gpu_ids 0,1,2,3 \
    --num_workers 4 \
    --n_feats 80 \
    --epochs_per_save 5 \
    --pretrained_ckpt_path ~/results/librispeech/ckpts/best.pth.tar \
