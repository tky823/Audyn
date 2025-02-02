## LSX

### Download Dataset

You can download LSX dataset by `audyn-download-lsx`.

```sh
data_root="./data"  # root directory to save .zip file.
lsx_root="${data_root}/lsx"
unpack=true  # unpack .zip or not
chunk_size=8192  # chunk size in byte to download

audyn-download-lsx \
root="${data_root}" \
lsx_root="${lsx_root}" \
unpack=${unpack} \
chunk_size=${chunk_size}
```
