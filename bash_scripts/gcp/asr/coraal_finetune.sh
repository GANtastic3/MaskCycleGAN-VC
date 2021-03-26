python -W ignore::UserWarning -m asr.main \
    --name coraal_finetune \
    --num_epochs 30 \
    --ckpt_path /home/results/librispeech_vanilla/ckpts/021_SpeechRecognitionModel.pth.tar \
    --coraal \
    --batch_size 10 \
    --continue_train \
