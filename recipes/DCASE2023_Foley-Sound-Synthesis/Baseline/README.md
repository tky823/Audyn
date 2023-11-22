# Baseline recipe of Foley Sound Synthesis task in DCASE 2023

This recipe reproduces the baseline system of Foley Sound Synthesis task in DCASE 2023.
The original implementation can be found at https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline.

## Stages

### Stage 0: Preprocess official development dataset

```sh
data="vqvae"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Preprocess UrbanSound8K

```sh
data="hifigan"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--data "${data}"
```

### Stage 2: Train VQVAE

```sh
data="vqvae"
train="vqvae"
model="vqvae"
optimizer="vqvae_ema"  # "vqvae"
lr_scheduler="none"
criterion="vqvae"

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--tag <TAG> \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr_scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 3: Save prior of VQVAE

```sh
data="vqvae"
train="prior"
model="vqvae"

vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--tag <TAG> \
--vqvae-checkpoint "${vqvae_checkpoint}" \
--data "${data}" \
--train "${train}" \
--model "${model}"
```

### Stage 4: Train PixelSNAIL

```sh
data="vqvae"
train="pixelsnail"
model="pixelsnail"
optimizer="pixelsnail"
lr_scheduler="none"
criterion="pixelsnail"

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

### Stage 5: Train HiFiGAN

```sh
data="hifigan"
train="hifigan"
model="hifigan_v1"
optimizer="hifigan"
lr_scheduler="hifigan"
criterion="hifigan"

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
