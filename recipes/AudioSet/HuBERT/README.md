# HuBERT using AudioSet-

## Stages

### Stage -1: Downloading dataset

**NOTE**: `yt-dlp` is required to download dataset.

```sh
. ./run.sh \
--stage -1 \
--stop-stage -1
```

### Stage 0: Preprocessing

```sh
data="hubert"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```
