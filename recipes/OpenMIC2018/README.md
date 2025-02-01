## OpenMIC2018

### Download Dataset

You can download OpenMIC2018 dataset by `audyn-download-openmic2018`.

```sh
data_root="./data"  # root directory to save .zip file.
openmic2018_root="${data_root}/openmic-2018"
unpack=true  # unpack .tgz or not
chunk_size=8192  # chunk size in byte to download

audyn-download-openmic2018 \
root="${data_root}" \
openmic2018_root="${openmic2018_root}" \
unpack=${unpack} \
chunk_size=${chunk_size}
```
