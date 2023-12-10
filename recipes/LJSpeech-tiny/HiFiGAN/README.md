# HiFi-GAN

## Stages

### Stage 0: Preprocess

```sh
dump_format="torch"  # webdataset

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--dump-format "${dump_format}"
```

### Stage 1: Train HiFi-GAN

```sh
dump_format="torch"

model="hifigan_tiny"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--dump-format "${dump_format}" \
--model "${model}"
```

If you resume training from a checkpoint,

```sh
dump_format="torch"

checkpoint=<PATH/TO/HIFIGAN/CHECKPOINT>  # e.g. exp/<TAG>/model/last.pth

model="hifigan_tiny"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--continue-from "${checkpoint}" \
--dump-format "${dump_format}" \
--model "${model}"
```
