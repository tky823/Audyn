# SSAST using AudioSet-tiny

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
data="ssast-tiny"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Training SSAST

To train SSAST, run the following command:

```sh
tag=<TAG>

data="ssast-tiny"
train="ssast-tiny"
model="multitask_ssast_patch_mask40-tiny"
optimizer="adam"
lr_scheduler="ssast"
criterion="ssast"

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
