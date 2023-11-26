# Baseline recipe of Foley Sound Synthesis task in DCASE 2023

This recipe reproduces the baseline system of Foley Sound Synthesis task in DCASE 2023.
The original implementation can be found at https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline.

## Models

### PixelSNAIL

Our implementation is based on https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline/blob/main/pixelsnail.py. However, we changed some details.

## Stages

### Stage 0: Preprocess official development dataset

```sh
data="baseline"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Preprocess test set

```sh
data="baseline"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--data "${data}"
```

### Stage 2: Preprocess UrbanSound8K

```sh
data="baseline"

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--data "${data}"
```

### Stage 3: Train VQVAE

```sh
data="baseline"
train="vqvae"
model="vqvae"
optimizer="vqvae_ema"  # "vqvae"
lr_scheduler="none"
criterion="vqvae"

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--tag <TAG> \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr_scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 4: Save prior of VQVAE

```sh
data="baseline"
train="prior"
model="vqvae"

vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth

. ./run.sh \
--stage 4 \
--stop-stage 4 \
--tag <TAG> \
--vqvae-checkpoint "${vqvae_checkpoint}" \
--data "${data}" \
--train "${train}" \
--model "${model}"
```

### Stage 5: Train PixelSNAIL

```sh
data="baseline"
train="pixelsnail"
model="pixelsnail"
optimizer="pixelsnail"
lr_scheduler="none"
criterion="pixelsnail"

. ./run.sh \
--stage 5 \
--stop-stage 5 \
--tag <TAG> \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr_scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 6: Train HiFi-GAN

```sh
data="baseline"
train="hifigan"
model="hifigan_v1"
optimizer="hifigan"
lr_scheduler="hifigan"
criterion="hifigan"

. ./run.sh \
--stage 6 \
--stop-stage 6 \
--tag <TAG> \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr_scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 7: Generate conditional audio samples

```sh
data="baseline"
test="baseline"
model="baseline"

pixelsnail_checkpoint=<PATH/TO/PIXELSNAIL/CHECKPOINT>  # e.g. exp/<TAG>/model/pixelsnail/last.pth
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth
hifigan_checkpoint=<PATH/TO/HIFIGAN/CHECKPOINT>  # e.g. exp/<TAG>/model/hifigan/last.pth

. ./run.sh \
--stage 7 \
--stop-stage 7 \
--tag <TAG> \
--pixelsnail-checkpoint "${pixelsnail_checkpoint}"
--vqvae-checkpoint "${vqvae_checkpoint}" \
--hifigan-checkpoint "${hifigan_checkpoint}" \
--data "${data}" \
--test "${test}" \
--model "${model}"
```
