# Poincare Embedding

## Stages

### Stage 0: Preprocessing

```sh
data="audioset_1"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Train Poincare Embedding

```sh
data="audioset_1"
model="poincare_embedding"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--data "${data}" \
--model "${model}"
```
