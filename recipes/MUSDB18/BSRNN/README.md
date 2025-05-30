# Band-split RNN (BSRNN)

## Stages

### Stage 0: Preprocess

```sh
data="musdb18-bass"  # "musdb18-bass", "musdb18-drums", "musdb18-other", or "musdb18-vocals"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Train BSRNN

```sh
data="musdb18-bass"
model="bsrnn-v7"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--model "${model}" \
--data "${data}"
```

If you resume training from a checkpoint,

```sh
checkpoint=<PATH/TO/BSRNN/CHECKPOINT>  # e.g. exp/<TAG>/model/last.pth

data="musdb18-bass"
model="bsrnn-v7"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--continue-from "${checkpoint}" \
--model "${model}" \
--data "${data}"
```
