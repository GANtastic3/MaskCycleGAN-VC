python -W ignore::UserWarning -m asr.main \
    --name librispeech_vanilla \
    --num_epochs 30 \
    --ckpt_path /home/results/librispeech_vanilla/ckpts/021_SpeechRecognitionModel.pth.tar \
    --continue_train \
