# Band-split RNN (BSRNN)

## Stages

### Stage 0: Preprocess

```sh
data="dnr-v2"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Train BSRNN

```sh
data="dnr-v2"
model="bsrnn_music-scale"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--data "${data}" \
--model "${model}"
```

If you resume training from a checkpoint,

```sh
checkpoint=<PATH/TO/BSRNN/CHECKPOINT>  # e.g. exp/<TAG>/model/last.pth

data="dnr-v2"
model="bsrnn_music-scale"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--continue-from "${checkpoint}" \
--data "${data}" \
--model "${model}"
```
