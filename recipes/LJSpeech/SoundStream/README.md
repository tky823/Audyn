# SoundStream using LJSpeech

## Stages

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

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag "${tag}"
```

To train SounsStream using official configs, set `train`, `model`, and `criterion` as follows:

```sh
tag=<TAG>

train="official_soundstream"
model="official_soundstream"
criterion="official_soundstream"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag "${tag}" \
--train "${train}" \
--model "${model}" \
--criterion "${criterion}"
```

### Stage 2: Reconstruction of speeches

```sh
tag=<TAG>
checkpoint=<PATH/TO/PRETRAINED/SOUNDSTREAM>

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--tag "${tag}" \
--checkpoint "${checkpoint}"
```

### Stage 3: Save quantized features as codebook indices

```sh
tag=<TAG>
checkpoint=<PATH/TO/PRETRAINED/SOUNDSTREAM>

train="save_quantized_features"

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--tag "${tag}" \
--checkpoint "${checkpoint}" \
--train "${train}"
```
