## FSD50K
Recipes using FSD50K dataset.

### Download dataset

```sh
data_root="./data"  # root directory to save .zip file.
fsd50k_root="${data_root}/FSD50K"
unpack=true  # unpack .zip or not
chunk_size=8192  # chunk size in byte to download

audyn-download-fsd50k \
root="${data_root}" \
fsd50k_root="${fsd50k_root}" \
unpack=${unpack} \
chunk_size=${chunk_size}
```
