python -m cycleGAN_VC3.generate \
    --name source_voc_6_target_coraal_ROC_se0_ag3_f_011 \
    --data_dir /home/ubuntu/data \
    --save_dir /home/ubuntu/results/cycleGAN_VC3/converted \
    --source_id 6 \
    --ckpt_path /home/ubuntu/results/cycleGAN_VC3/source_voc_6_target_coraal_ROC_se0_ag3_f_011/ckpts/1000_generator_A2B.pth.tar \
    --load_epoch 5500 \
    --gpu_ids 3