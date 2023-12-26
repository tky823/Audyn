# Baseline recipe of Foley Sound Synthesis task in DCASE 2023

This recipe reproduces the baseline system of Foley Sound Synthesis task in DCASE 2023.
The original implementation can be found at https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline.

## Models

### PixelSNAIL

Our implementation is based on https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline/blob/main/pixelsnail.py. However, we changed some details.

## Stages

### Stage 0: Preprocess UrbanSound8K (optional)

```sh
dump_format="torch"

data="baseline"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--dump-format "${dump_format}" \
--data "${data}"
```

### Stage 1: Train HiFi-GAN (optional)

```sh
dump_format="torch"

system="defaults"  # "cuda", "cuda_ddp", "cuda_amp", "cuda_ddp_amp"
data="baseline"
train="hifigan"
model="hifigan_v1"
optimizer="hifigan"
lr_scheduler="hifigan"
criterion="hifigan"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--dump-format "${dump_format}" \
--system "${system}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr_scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 2: Preprocess official development dataset

```sh
dump_format="torch"

data="baseline"

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--dump-format "${dump_format}" \
--data "${data}"
```

### Stage 3: Preprocess test set

```sh
dump_format="torch"

data="baseline"

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--dump-format "${dump_format}" \
--data "${data}"
```

### Stage 4: Train VQVAE

If you choose `vqvae_ema` as `optimizer`, you cannot use mixed precision (`cuda_amp` and `cuda_ddp_amp`).

If you want to use RVQVAE, set `data="baseline_rvqvae"`,  `model="rvqvae"`, and `criterion="rvqvae"`. `optimizer="vqvae_ema"` is also supported even in this case.

```sh
dump_format="torch"

system="defaults"  # "cuda", "cuda_ddp"
data="baseline"
train="vqvae"
model="vqvae"
optimizer="vqvae_ema"  # "vqvae"
lr_scheduler="none"
criterion="vqvae"

. ./run.sh \
--stage 4 \
--stop-stage 4 \
--tag <TAG> \
--dump-format "${dump_format}" \
--system "${system}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr_scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

If you want to use pretrained HiFi-GAN as a vocoder, please set `--train vqvae+pretrained_hifigan`.
In addition, `hifigan_checkpoint` is required.
In this setting, parameters in HiFi-GAN are not updated by the optimizer.

```sh
dump_format="torch"

hifigan_checkpoint=<PATH/TO/PRETRAINED/HIFIGAN/CHECKPOINT>

system="defaults"  # "cuda", "cuda_ddp"
data="baseline"
train="vqvae+pretrained_hifigan"
model="vqvae"
optimizer="vqvae_ema"  # "vqvae"
lr_scheduler="none"
criterion="vqvae"

. ./run.sh \
--stage 4 \
--stop-stage 4 \
--tag <TAG> \
--dump-format "${dump_format}" \
--hifigan-checkpoint "${hifigan_checkpoint}" \
--system "${system}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr_scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 5: Save prior of VQVAE

```sh
dump_format="torch"

system="defaults"  # "cuda"
data="baseline"
train="prior"
model="vqvae"

vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth

. ./run.sh \
--stage 5 \
--stop-stage 5 \
--tag <TAG> \
--dump-format "${dump_format}" \
--vqvae-checkpoint "${vqvae_checkpoint}" \
--system "${system}" \
--data "${data}" \
--train "${train}" \
--model "${model}"
```

### Stage 6: Train PixelSNAIL

You cannot use mixed precision (`cuda_amp` and `cuda_ddp_amp`) for training of PixelSNAIL due to numerical instability.

```sh
dump_format="torch"

system="defaults"  # "cuda", "cuda_ddp"
data="baseline"
train="pixelsnail"
model="pixelsnail"
optimizer="pixelsnail"
lr_scheduler="none"
criterion="pixelsnail"

. ./run.sh \
--stage 6 \
--stop-stage 6 \
--tag <TAG> \
--dump-format "${dump_format}" \
--system "${system}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr_scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 7: Generate conditional audio samples

```sh
dump_format="torch"

system="defaults"  # "cuda"
data="baseline"
test="baseline"
model="baseline"

pixelsnail_checkpoint=<PATH/TO/PIXELSNAIL/CHECKPOINT>  # e.g. exp/<TAG>/model/pixelsnail/last.pth
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth
hifigan_checkpoint=<PATH/TO/HIFIGAN/CHECKPOINT>  # e.g. exp/<TAG>/model/hifigan/last.pth

. ./run.sh \
--stage 7 \
--stop-stage 7 \
--tag <TAG> \
--dump-format "${dump_format}" \
--pixelsnail-checkpoint "${pixelsnail_checkpoint}" \
--vqvae-checkpoint "${vqvae_checkpoint}" \
--hifigan-checkpoint "${hifigan_checkpoint}" \
--system "${system}" \
--data "${data}" \
--test "${test}" \
--model "${model}"
```
