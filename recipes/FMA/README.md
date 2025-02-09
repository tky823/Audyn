## Free Music Archive
Recipes using Free Music Archive (FMA) dataset.

### Download Dataset

You can download FMA dataset by `audyn-download-fma`.

```sh
type="small"  # for FMA-small
# type="medium"  # for FMA-medium
# type="large"  # for FMA-large
# type="full"  # for FMA-full

data_root="./data"  # root directory to save .zip file.
fma_root="${data_root}/FMA/${type}"
unpack=true  # unpack .zip or not
chunk_size=8192  # chunk size in byte to download

audyn-download-fma \
type="${type}" \
root="${data_root}" \
fma_root="${fma_root}" \
unpack=${unpack} \
chunk_size=${chunk_size}
```
