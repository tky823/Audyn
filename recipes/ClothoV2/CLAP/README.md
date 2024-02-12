# Recipe of CLAP using ClothoV2

## Stages

### Stage 0: Preprocess dataset

```sh
dump_format="torch"

data="clotho-v2_mel64"  # or "clotho-v2_mel128", "clotho-v2_mel256", "clotho-v2_bert-ssast"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--dump-format "${dump_format}" \
--data "${data}"
```

### Stage 1: Train text tower

**NOTE**: If you use pretrained BERT and SSAST, you can skip this stage.

```sh
dump_format="torch"

data="clotho-v2_mel64"  # or "clotho-v2_mel128", "clotho-v2_mel256"
train="text_transformer"
model="text_transformer"
optimizer="text_transformer"
lr_scheduler="text_transformer"
criterion="text_mlm"

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

### Stage 2: Train Audio tower

**NOTE**: If you use pretrained BERT and SSAST, you can skip this stage.

```sh
dump_format="torch"

data="clotho-v2_mel64"  # or "clotho-v2_mel128", "clotho-v2_mel256"
train="audio_transformer"
model="audio_transformer"
optimizer="audio_transformer"
lr_scheduler="audio_transformer"
criterion="audio_mpm"

. ./run.sh \
--stage 2 \
--stop-stage 2 \
--tag <TAG> \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 3: Train CLAP

**NOTE**: If you use pretrained BERT and SSAST, you don't have to set neither of `text_tower_checkpoint` and `audio_tower_checkpoint`.

```sh
dump_format="torch"

text_tower_checkpoint="<PATH/TO/PRETRAINED/TEXT/TOWER>"
audio_tower_checkpoint="<PATH/TO/PRETRAINED/AUDIO/TOWER>"

data="clotho-v2_mel64"  # or "clotho-v2_mel128", "clotho-v2_mel256", "clotho-v2_bert-ssast"
train="clap" # or "clap_bert-ssast"
model="clap_cls"  # or "clap_pool", "clap_bert-ssast"
criterion="clap"
optimizer="clap"
lr_scheduler="clap"

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--tag <TAG> \
--dump-format "${dump_format}" \
--text-tower-checkpoint "${text_tower_checkpoint}" \
--audio-tower-checkpoint "${audio_tower_checkpoint}" \
--data "${data}" \
--train "${train}" \
--model "${model}" \
--optimizer "${optimizer}" \
--lr-scheduler "${lr_scheduler}" \
--criterion "${criterion}"
```

### Stage 4: Save text and audio embeddings of CLAP

```sh
dump_format="torch"

clap_checkpoint="<PATH/TO/PRETRAINED/CLAP>"

data="clotho-v2_mel64"  # or "clotho-v2_mel128", "clotho-v2_mel256"
test="save_embeddings"
model="clap_cls"  # or "clap_pool"

. ./run.sh \
--stage 4 \
--stop-stage 4 \
--tag <TAG> \
--dump-format "${dump_format}" \
--clap-checkpoint "${clap_checkpoint}" \
--data "${data}" \
--test "${test}" \
--model "${model}"
```

### Stage 5: Evaluate retrieval results

Only `system=cpu` is supported in this stage.

```sh
dump_format="torch"

system="cpu"
data="clotho-v2_mel64"  # or "clotho-v2_mel128", "clotho-v2_mel256"
test="clap"
model="clap_cls"  # or "clap_pool"

. ./run.sh \
--stage 5 \
--stop-stage 5 \
--tag <TAG> \
--dump-format "${dump_format}" \
--system "${system}" \
--data "${data}" \
--test "${test}" \
--model "${model}"
```
