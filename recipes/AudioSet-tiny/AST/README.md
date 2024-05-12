# AST using AudioSet-tiny

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
data="ast-tiny"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Training AST

To train AST, run the following command:

```sh
tag=<TAG>

data="ast-tiny"
train="ast-tiny"
model="ast-tiny"
optimizer="adam"
lr_scheduler="ast"
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
