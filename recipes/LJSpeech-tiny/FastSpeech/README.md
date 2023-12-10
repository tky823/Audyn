# FastSpeech

## Stages

### Stage 0: Preprocess

```sh
dump_format="torch"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--dump-format "${dump_format}"
```

### Stage 1: Train FastSpeech

```sh
dump_format="torch"

model="fastspeech"

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

checkpoint=<PATH/TO/FASTSPEECH/CHECKPOINT>  # e.g. exp/<TAG>/model/last.pth

model="fastspeech"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--continue-from "${checkpoint}" \
--dump-format "${dump_format}" \
--model "${model}"
```
