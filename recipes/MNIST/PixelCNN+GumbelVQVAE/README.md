# PixelCNN + GumbelVQVAE recipe using MNIST

## Stages

### Stage 1: train GumbelVQVAE

```sh
dump_format="torch"

data="gumbel-vqvae"
train="gumbel-vqvae"
model="gumbel-vqvae"
optimizer="vqvae"
lr_scheduler="vqvae"
criterion="gumbel-vqvae_melspectrogram"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

If you resume training from a checkpoint,

```sh
dump_format="torch"

data="gumbel-vqvae"
train="gumbel-vqvae"
model="gumbel-vqvae"
optimizer="vqvae"
lr_scheduler="vqvae"
criterion="gumbel-vqvae_melspectrogram"
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--tag <TAG> \
--continue-from "${vqvae_checkpoint}" \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```
