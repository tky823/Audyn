# WaveNeXt

## Stages

### Stage 0: Preprocess

```sh
dump_format="torch"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--dump-format "${dump_format}"
```

### Stage 1: Train WaveNeXt

```sh
dump_format="torch"

model="wavenext_tiny"

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

checkpoint=<PATH/TO/WAVENEXT/CHECKPOINT>  # e.g. exp/<TAG>/model/last.pth

model="wavenext_tiny"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--continue-from "${checkpoint}" \
--dump-format "${dump_format}" \
--model "${model}"
```
