## Free Music Archive (small)
Recipes using Free Music Archive small (FMA-small) dataset.

### Download Dataset

You can download FMA-small dataset by `audyn-download-fma`.

```sh
type="small"  # for FMA-small

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
