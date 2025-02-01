## DnR-V2

### Download Dataset

You can download DnR-V2 dataset by `audyn-download-dnr`.

```sh
data_root="./data"  # root directory to save .tar.gz file.
dnr_root="${data_root}/DnR-V2"
version=2
unpack=true  # unpack .tar.gz or not
chunk_size=8192  # chunk size in byte to download

audyn-download-dnr \
root="${data_root}" \
dnr_root="${dnr_root}" \
version=${version} \
unpack=${unpack} \
chunk_size=${chunk_size}
```
