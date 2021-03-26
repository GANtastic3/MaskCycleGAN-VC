python -m cycleGAN_VC3.generate \
    --name source_voc_26_target_coraal_PRV_se0_ag2_m_01 \
    --data_dir /home/ubuntu/data \
    --save_dir /home/ubuntu/results/cycleGAN_VC3/converted \
    --source_id 26 \
    --ckpt_path /home/ubuntu/results/cycleGAN_VC3/source_voc_26_target_coraal_PRV_se0_ag2_m_01/ckpts/1250_generator_A2B.pth.tar \
    --load_epoch 1250 \
    --gpu_ids 0