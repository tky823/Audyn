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
