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
dump_format="birdclef2024"

data="birdclef2024"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--dump-format "${dump_format}" \
--data "${data}"
```

### Stage 1: Training baseline model

To train baseline model, run the following command:

```sh
tag=<TAG>

dump_format="birdclef2024"

data="birdclef2024"
train="birdclef2024baseline"
model="birdclef2024baseline"
optimizer="adam"
lr_scheduler="none"
criterion="birdclef2024"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag "${tag}" \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 2: Inference by baseline model

To infer by baseline model, run the following command:

```sh
tag=<TAG>

checkpoint=<PATH/TO/TRAINED/MODEL>

dump_format="birdclef2024"

data="birdclef2024"
train="birdclef2024baseline"
test="birdclef2024baseline"
model="birdclef2024baseline"

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--tag "${tag}" \
--checkpoint "${checkpoint}" \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--test "${test}" \
--model "${model}"
```
