# Band-split RoFormer (BSRoFormer)

## Stages

### Stage 0: Preprocess

```sh
data="dnr-v2"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Train BSRoFormer

```sh
data="dnr-v2"
model="bsroformer_music-scale"
criterion="l1snr"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--data "${data}" \
--model "${model}" \
--criterion "${criterion}"
```

If you resume training from a checkpoint,

```sh
checkpoint=<PATH/TO/BSROFORMER/CHECKPOINT>  # e.g. exp/<TAG>/model/last.pth

data="dnr-v2"
model="bsroformer_music-scale"
criterion="l1snr"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--continue-from "${checkpoint}" \
--data "${data}" \
--model "${model}" \
--criterion "${criterion}"
```
