python -m cycleGAN_VC3.generate \
    --name source_voc_13_target_coraal_ATL_se0_ag1_f_01 \
    --data_dir /home/ubuntu/data \
    --save_dir /home/ubuntu/results/cycleGAN_VC3/converted \
    --source_id 13 \
    --ckpt_path /home/ubuntu/results/cycleGAN_VC3/source_voc_13_target_coraal_ATL_se0_ag1_f_01/ckpts/5500_generator_A2B.pth.tar \
    --load_epoch 5500 \
    --gpu_ids 3