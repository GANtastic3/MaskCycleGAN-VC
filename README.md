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

