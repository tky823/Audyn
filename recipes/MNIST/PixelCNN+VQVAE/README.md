# PixelCNN + VQVAE recipe using MNIST

## Stages

### Stage 1: train VQVAE

```sh
dump_format="torch"

train="vqvae_ema"  # vqvae
model="vqvae"
optimizer="vqvae_ema"  # vqvae
lr_scheduler="vqvae_ema"  # vqvae
criterion="vqvae"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--dump-format "${dump_format}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

If you resume training from a checkpoint,

```sh
dump_format="torch"

train="vqvae_ema"  # vqvae
model="vqvae"
optimizer="vqvae_ema"  # vqvae
lr_scheduler="vqvae_ema"  # vqvae
criterion="vqvae"
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae_ema/last.pth

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--continue-from "${vqvae_checkpoint}" \
--dump-format "${dump_format}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 2: save priors in latent space

```sh
dump_format="torch"

train="prior"
model="vqvae"
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae_ema/last.pth

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--tag <TAG> \
--dump-format "${dump_format}" \
--vqvae-checkpoint "${vqvae_checkpoint}" \
--train "${train}" \
--model "${model}"
```

### Stage 3: train PixelCNN

```sh
dump_format="torch"

train="pixelcnn"
model="pixelcnn"
optimizer="pixelcnn"
lr_scheduler="pixelcnn"
criterion="pixelcnn"

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--tag <TAG> \
--dump-format "${dump_format}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```


### Stage 4: generate images using PixelCNN + VQVAE

```sh
dump_format="torch"

test="pixelcnn+vqvae"
model="pixelcnn+vqvae"
pixelcnn_checkpoint=<PATH/TO/PIXELCNN/CHECKPOINT>  # e.g. exp/<TAG>/model/pixelcnn/last.pth
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae_ema/last.pth

. ./run.sh \
--stage 4 \
--stop-stage 4 \
--pixelcnn-checkpoint "${pixelcnn_checkpoint}" \
--vqvae-checkpoint "${vqvae_checkpoint}" \
--tag <TAG> \
--dump-format "${dump_format}" \
--test "${test}" \
--model "${model}"
```
