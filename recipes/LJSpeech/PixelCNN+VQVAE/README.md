# PixelCNN + VQVAE recipe using LJSpeech

## Stages

### Stage 0: preprocess

```sh
. ./run.sh \
--stage 0 \
--stop-stage 0
```

### Stage 1: train VQVAE

```sh
train="vqvae"  # vqvae_ema
model="vqvae"
optimizer="vqvae"  # vqvae_ema
lr_scheduler="vqvae"  # vqvae_ema
criterion="vqvae"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

If you resume training from a checkpoint,

```sh
train="vqvae"  # vqvae_ema
model="vqvae"
optimizer="vqvae"  # vqvae_ema
lr_scheduler="vqvae"  # vqvae_ema
criterion="vqvae"
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--continue-from "${vqvae_checkpoint}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 2: save priors in latent space

```sh
train="prior"
model="vqvae"
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--tag <TAG> \
--vqvae-checkpoint "${vqvae_checkpoint}" \
--train "${train}" \
--model "${model}"
```

### Stage 3: train PixelCNN

```sh
train="pixelcnn"
model="pixelcnn"
optimizer="pixelcnn"
lr_scheduler="pixelcnn"
criterion="pixelcnn"

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--tag <TAG> \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 4: generate melspectrogram using PixelCNN + VQVAE

```sh
test="pixelcnn+vqvae"
model="pixelcnn+vqvae"
pixelcnn_checkpoint=<PATH/TO/PIXELCNN/CHECKPOINT>  # e.g. exp/<TAG>/model/pixelcnn/last.pth
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth

. ./run.sh \
--stage 4 \
--stop-stage 4 \
--pixelcnn-checkpoint "${pixelcnn_checkpoint}" \
--vqvae-checkpoint "${vqvae_checkpoint}" \
--tag <TAG> \
--test "${test}" \
--model "${model}"
```
