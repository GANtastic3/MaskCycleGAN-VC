python -W ignore::UserWarning -m asr.main \
    --name coraal_small_finetune \
    --data_dir ~/data/datasets \
    --save_dir ~/data/results \
    --coraal \
    --small_dataset \
    --num_epochs 50 \
    --batch_size 10 \
    --gpu_ids 0 \
    --num_workers 1 \
