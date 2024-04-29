# PaSST using AudioSet-tiny

## Stages

### Stage -1: Downloading dataset

At this stage, we try to download all audio samples from YouTube, which takes so long time.
See ``recipes/AudioSet/_common/README.md`` for the details.

**NOTE**: `yt-dlp` is required to download dataset.

```sh
. ./run.sh \
--stage -1 \
--stop-stage -1
```

### Stage 0: Preprocessing

```sh
data="passt"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Training PaSST

To train PaSST using structured patchout, run the following command:

```sh
tag=<TAG>

data="passt"
train="passt"
model="passt-stride10-struct"
optimizer="passt"
lr_scheduler="passt"
criterion="audioset"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag "${tag}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

Instead, you can train PaSST using unstructured patchout by the following command:

```sh
tag=<TAG>

data="passt"
train="passt"
model="passt-stride10-unstruct"
optimizer="passt"
lr_scheduler="passt"
criterion="audioset"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag "${tag}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```
