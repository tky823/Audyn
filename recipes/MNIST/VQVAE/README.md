# VQVAE recipe using MNIST

## Stages

### Stage 1: train VQVAE

```sh
data="vqvae"
train="vqvae"
test="vqvae"
model="vqvae"
optimizer="vqvae"
lr_scheduler="vqvae"
criterion="vqvae"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--data "${data}" \
--train "${train}" \
--test "${test}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 2: save priors in latent space

```sh
data="vqvae"
train="prior"
model="vqvae"
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--tag <TAG> \
--data "${data}" \
--vqvae-checkpoint "${vqvae_checkpoint}" \
--train "${train}" \
--model "${model}"
```

### Stage 3: train PixelCNN

```sh
data="vqvae"
train="pixelcnn"
test="pixelcnn"
model="pixelcnn"
optimizer="pixelcnn"
lr_scheduler="pixelcnn"
criterion="pixelcnn"

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--tag <TAG> \
--data "${data}" \
--train "${train}" \
--test "${test}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```
