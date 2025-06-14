# Band-split RoFormer (BSRoFormer)

## Stages

### Stage 0: Preprocess

```sh
data="musdb18hq-bass"  # "musdb18hq-bass", "musdb18hq-drums", "musdb18hq-other", or "musdb18hq-vocals"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Train BSRoFormer

```sh
data="musdb18hq-bass"  # "musdb18hq-bass", "musdb18hq-drums", "musdb18hq-other", or "musdb18hq-vocals"
model="bsroformer_v7"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--data "${data}" \
--model "${model}"
```

If you resume training from a checkpoint,

```sh
checkpoint=<PATH/TO/BSROFORMER/CHECKPOINT>  # e.g. exp/<TAG>/model/last.pth

data="musdb18hq-bass"  # "musdb18hq-bass", "musdb18hq-drums", "musdb18hq-other", or "musdb18hq-vocals"
model="bsroformer_v7"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--continue-from "${checkpoint}" \
--data "${data}" \
--model "${model}"
```
