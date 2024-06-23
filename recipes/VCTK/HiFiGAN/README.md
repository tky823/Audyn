# HiFi-GAN using VCTK

## Stages

### Stage -1: Downloading dataset

```sh
. ./run.sh \
--stage -1 \
--stop-stage -1
```

### Stage 0: Preprocessing

```sh
data="hifigan_vctk"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Train HiFi-GAN

```sh
data="hifigan_vctk"
model="hifigan_v1"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--data "${data}" \
--model "${model}"
```
