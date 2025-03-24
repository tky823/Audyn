# PixelCNN + RVQVAE recipe using CIFAR100

## Stages

### Stage 1: train RVQVAE

```sh
dump_format="torch"

train="rvqvae"
model="rvqvae"
optimizer="vqvae_ema"  # vqvae
lr_scheduler="vqvae_ema"  # vqvae
criterion="rvqvae_melspectrogram"

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

train="rvqvae"
model="rvqvae"
optimizer="vqvae_ema"  # vqvae
lr_scheduler="vqvae_ema"  # vqvae
criterion="rvqvae_melspectrogram"
vqvae_checkpoint=<PATH/TO/VQVAE/CHECKPOINT>  # e.g. exp/<TAG>/model/vqvae/last.pth

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
