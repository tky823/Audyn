# BirdCLEF2024

[Baseline recipe](https://www.kaggle.com/code/awsaf49/birdclef24-kerascv-starter-train) of [BirdCLEF2024](https://www.kaggle.com/competitions/birdclef-2024) challenge.

## Stages

### Stage -1: Downloading dataset

Download dataset and place it as `../data/birdclef-2024.zip`.
Then, unzip the file.

```sh
recipes/BirdCLEF2024/
|- data/
    |- birdclef-2024.zip
    |- birdclef-2024/
        |- eBird_Taxonomy_v2021.csv
        |- train_metadata.csv
        |- sample_submission.csv
        |- train_audio/
        |- test_soundscapes/
        |- unlabeled_soundscapes/
```

### Stage 0: Preprocessing

```sh
data="birdclef2024"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```
