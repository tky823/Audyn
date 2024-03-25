# SoundStream using LJSpeech

## Stages

### Stage -1: Downloading dataset

```sh
. ./run.sh \
--stage 0 \
--stop-stage 0
```

### Stage 0: Preprocessing

```sh
data="soundstream"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Training SoundStream

By default, codebooks in RVQ are updated by exponential moving average.
If the codebook usage is less than `reset_ath`, the codebook is randomly replaced with an encoded feature in the current batch.
Since the absolute usage depends on the `batch_size`  (defined in `conf/train`), `slice_length` (defined in `conf/data`), `codebook_size` (defined in `conf/data`), and down sampling rate of the model, please set appropriate value on your own.

e.g.) `batch_size=32`, `slice_length=72000`, `codebook_size=1024`, and down sampling rate of SoundStream is `320` (i.e. 24kHz -> 75Hz), the ideal usage of each codebook is (32 * 72000) / (320 * 1024) = 7.03.

```yaml
generator:
  ...
  - name: ema
    optimizer:
      _target_: audyn.optim.optimizer.ExponentialMovingAverageCodebookOptimizer
      smooth: 0.99
      reset_step: 1
      reset_var: 0
      reset_ath: 2
      reset_source: batch
      reset_scope: all
    modules:
      - vector_quantizer
...
```

To train SoundStream, run the following command:

```sh
tag=<TAG>

data="soundstream"
train="soundstream"
model="soundstream"
optimizer="soundstream"
lr_scheduler="soundstream"
criterion="soundstream"

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

To train SounsStream using official configs, set `train`, `model`, and `criterion` as follows:

```sh
tag=<TAG>

data="soundstream"
train="official_soundstream"
model="official_soundstream"
optimizer="soundstream"
lr_scheduler="soundstream"
criterion="official_soundstream"

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

### Stage 2: Reconstruction of speeches

```sh
tag=<TAG>
checkpoint=<PATH/TO/PRETRAINED/SOUNDSTREAM>  # e.g. ${exp_root}/${tag}/model/soundstream/last.pth

data="soundstream"
test="soundstream_reconstruction"
model="soundstream_reconstructor"

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--tag "${tag}" \
--checkpoint "${checkpoint}" \
--data "${data}" \
--test "${test}" \
--model "${model}"
```

### Stage 3: Save quantized features as codebook indices

```sh
tag=<TAG>
checkpoint=<PATH/TO/PRETRAINED/SOUNDSTREAM>  # e.g. ${exp_root}/${tag}/model/soundstream/last.pth

data="soundstream"
train="save_quantized_features"
model="soundstream"

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--tag "${tag}" \
--checkpoint "${checkpoint}" \
--data "${data}" \
--train "${train}" \
--model "${model}"
```

### Stage 4: Convert SoundStream to SoundStreamFirstStageDecoder

```sh
tag=<TAG>
checkpoint=<PATH/TO/PRETRAINED/SOUNDSTREAM>  # e.g. ${exp_root}/${tag}/model/soundstream/last.pth

data="soundstream"
train="convert_soundstream"

. ./run.sh \
--stage 4 \
--stop-stage 4 \
--tag "${tag}" \
--checkpoint "${checkpoint}" \
--data "${data}" \
--train "${train}"
```

Then, the converted model is saved as `${exp_root}/${tag}/model/soundstream_first_stage_decoder/*.pth`.

### Stage 5: Training of SoundStream-TTS

```sh
tag=<TAG>
feat_to_wave_checkpoint=<PATH/TO/PRETRAINED/SOUNDSTREAM>  # e.g. ${exp_root}/${tag}/model/soundstream_first_stage_decoder/last.pth

data="soundstream"
train="valle+pretrained_soundstream"
model="valle"
optimizer="valle"
lr_scheduler="valle"
criterion="valle"

. ./run.sh \
--stage 5 \
--stop-stage 5 \
--tag "${tag}" \
--feat-to-wave-checkpoint "${feat_to_wave_checkpoint}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 6: Synthesis of speeches

```sh
tag=<TAG>
text_to_feat_checkpoint=<PATH/TO/PRETRAINED/VALLE>  # e.g. ${exp_root}/${tag}/model/valle/last.pth
feat_to_wave_checkpoint=<PATH/TO/PRETRAINED/SOUNDSTREAM>  # e.g. ${exp_root}/${tag}/model/soundstream_first_stage_decoder/last.pth

data="soundstream"
test="valle_tts"
model="valle_tts"

. ./run.sh \
--stage 6 \
--stop-stage 6 \
--tag "${tag}" \
--text-to-feat-checkpoint "${text_to_feat_checkpoint}" \
--feat-to-wave-checkpoint "${feat_to_wave_checkpoint}" \
--data "${data}" \
--test "${test}" \
--model "${model}"
```
