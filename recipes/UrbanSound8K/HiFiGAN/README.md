# HiFi-GAN recipe of UrbanSound8K

This recipe aims at training of universal vocoder using UrbanSound8K dataset.

## Stages

### Stage -1: Download dataset

```sh
. ./run.sh \
--stage -1 \
--stop-stage -1
```

### Stage 0: Preprocess

```sh
. ./run.sh \
--stage 0 \
--stop-stage 0
```

### Stage 1: Train HiFi-GAN

```sh
model="hifigan_v1"  # or "hifigan_v2", "hifigan_v3"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--model "${model}"
```
