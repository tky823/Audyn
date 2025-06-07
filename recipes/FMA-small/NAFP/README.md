# Neural Audio Fingerprinting usine FMA-small

## Stages

### Stage -1: Downloading dataset

Follow description by running

```sh
. ./run.sh \
--stage -1 \
--stop-stage -1
```

### Stage 0: Preprocessing

```sh
data="fma-small"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```
