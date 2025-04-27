# Poincare Embedding

In this recipe, you can reproduce [Poincaré embeddings for learning hierarchical representations](https://arxiv.org/pdf/1705.08039).

## Stages

### Stage 0: Preprocessing

```sh
data="wordnet-mammal_1"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--data "${data}"
```

### Stage 1: Train Poincare Embedding

```sh
data="wordnet-mammal_1"
model="poincare_embedding"

. ./run.sh \
--stage 1 \
--stop-stage 1 \
--data "${data}" \
--model "${model}"
```
