# MaskCycleGAN-VC
Implementation of Kaneko et al.'s MaskCycleGAN-VC model for non-parallel voice conversion.

## VCC2018 Dataset

The authors of the paper used the dataset from the Spoke task of [Voice Conversion Challenge 2018 (VCC2018)](https://datashare.ed.ac.uk/handle/10283/3061). This is a dataset of non-parallel utterances from 6 male and 6 female speakers. Each speaker utters approaximately 80 sentences.

Download the dataset from the command line.
```
wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip?sequence=2&isAllowed=y
```

Unzip the dataset file.
```
apt-get install unzip
unzip vcc2018_database_training.zip?sequence=2
```

## Data Preprocessing

To expedite training, preprocess the dataset by converting waveforms to melspectograms, then save the spectrograms as pickle files `<speaker_id>normalized.pickle` and normalization statistics (mean, std) as npz files `<speaker_id>_norm_stats.npz`. We convert waveforms to spectrograms using the [melgan vocoder](https://github.com/descriptinc/melgan-neurips) so that we can decode voice converted spectrograms to waveform during inference.

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018_training \
  --preprocessed_data_directory vcc2018_training_preprocessed \
  --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2
```
