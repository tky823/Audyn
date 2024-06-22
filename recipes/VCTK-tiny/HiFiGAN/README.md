# HiFi-GAN using VCTK-tiny

## Stages

### Stage -1: Downloading dataset

```sh
. ./run.sh \
--stage -1 \
--stop-stage -1
```

### Stage 0: Preprocessing

```sh
data="hifigan_vctk-tiny"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```
