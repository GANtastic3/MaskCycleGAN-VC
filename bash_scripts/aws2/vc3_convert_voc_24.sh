python -m cycleGAN_VC3.generate \
    --name source_voc_24_target_coraal_DCB_se1_ag2_f_0 \
    --data_dir /home/ubuntu/data \
    --save_dir /home/ubuntu/results/cycleGAN_VC3/converted \
    --source_id 24 \
    --ckpt_path /home/ubuntu/results/cycleGAN_VC3/source_voc_24_target_coraal_DCB_se1_ag2_f_0/ckpts/1250_generator_A2B.pth.tar \
    --load_epoch 1250 \
    --gpu_ids 0