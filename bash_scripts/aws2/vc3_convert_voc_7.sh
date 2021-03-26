python -m cycleGAN_VC3.generate \
    --name source_voc_7_target_coraal_PRV_se0_ag1_m_02 \
    --data_dir /home/ubuntu/data \
    --save_dir /home/ubuntu/results/cycleGAN_VC3/converted \
    --source_id 7 \
    --ckpt_path /home/ubuntu/results/cycleGAN_VC3/source_voc_7_target_coraal_PRV_se0_ag1_m_02/ckpts/1750_generator_A2B.pth.tar \
    --load_epoch 2250 \
    --gpu_ids 0