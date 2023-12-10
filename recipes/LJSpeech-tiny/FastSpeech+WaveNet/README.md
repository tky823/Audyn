# FastSpeech + WaveNet

## Stages

### Stage 0 & 1
Please perform `run.sh` in `../FastSpeech` and `../WaveNet` recipes.

### Stage 2: Synthesize

```sh
dump_format="torch"

text_to_feat_checkpoint=<PATH/TO/FASTSPEECH/CHECKPOINT>  # ../FastSpeech/exp/<TAG>/model/last.pth
feat_to_wave_checkpoint=<PATH/TO/WAVENET/CHECKPOINT>  # ../WaveNet/exp/<TAG>/model/last.pth

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--dump-format "${dump_format}" \
--text-to-feat-checkpoint "${text_to_feat_checkpoint}" \
--feat-to-wave-checkpoint "${feat_to_wave_checkpoint}"
```
