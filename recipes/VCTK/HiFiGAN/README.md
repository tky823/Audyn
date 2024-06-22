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
data="hifigan"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```
