# Conv-TasNet using WSJ0-2mix

## Stages

### Stage -1: Prepare dataset

Under `${wsj0_2mix_root}`, place dataset as follows:

```sh
${wsj0_2mix_root}
|- tr/
|  |- scaling.mat
|  |- s1/
|  |  |- 01aa010b_0.97482_209a010p_-0.97482.wav
|  |  ...
|  |- s2/
|  |  |- 01aa010b_0.97482_209a010p_-0.97482.wav
|  |  ...
|  |- mix/
|     |- 01aa010b_0.97482_209a010p_-0.97482.wav
|     ...
|- cv/
|  |- scaling.mat
|  |- s1/
|  |- s2/
|  |- mix/
|- tt/
   |- scaling.mat
   |- s1/
   |- s2/
   |- mix/
```

By default, `${wsj0_2mix_root}` is set to `../data/wsj0-mix/2speakers/wav8k/min`.
