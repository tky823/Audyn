# Recipe of CLAP using ClothoV2

## Stages

### Stage 0: Preprocess dataset

```sh
dump_format="torch"

data="clotho-v2_mel64"  # or "clotho-v2_mel128", "clotho-v2_mel256"

. ./run.sh \
--stage 0 \
--stop-stage 0 \
--dump-format "${dump_format}" \
--data "${data}"
```

### Stage 3: Train CLAP

```sh
dump_format="torch"

data="clotho-v2_mel64"  # or "clotho-v2_mel128", "clotho-v2_mel256"
train="clap"
model="clap_cls"  # or "clap_pool"

. ./run.sh \
--stage 3 \
--stop-stage 3 \
--tag <TAG> \
--dump-format "${dump_format}" \
--data "${data}" \
--train "${train}" \
--model "${model}"
```

### Stage 4: Save text and audio embeddings of CLAP

```sh
dump_format="torch"

checkpoint="<PATH/TO/PRETRAINED/CLAP>"

data="clotho-v2_mel64"  # or "clotho-v2_mel128", "clotho-v2_mel256"
test="save_embeddings"
model="clap_cls"  # or "clap_pool"

. ./run.sh \
--stage 4 \
--stop-stage 4 \
--tag <TAG> \
--dump-format "${dump_format}" \
--checkpoint "${checkpoint}" \
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
