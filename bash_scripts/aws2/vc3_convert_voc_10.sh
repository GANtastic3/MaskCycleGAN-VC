python -m cycleGAN_VC3.generate \
    --name source_voc_10_target_coraal_PRV_se0_ag2_f_03 \
    --data_dir /home/ubuntu/data \
    --save_dir /home/ubuntu/results/cycleGAN_VC3/converted \
    --source_id 10 \
    --ckpt_path /home/ubuntu/results/cycleGAN_VC3/source_voc_10_target_coraal_PRV_se0_ag2_f_03/ckpts/6000_generator_A2B.pth.tar \
    --load_epoch 6000 \
    --gpu_ids 1