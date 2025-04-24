## Song describer dataset (SDD)

### Download Dataset

You can download SDD by `audyn-download-song-describer`.

```sh
data_root="./data"  # root directory to save .zip file.
song_describer_root="${data_root}/SongDescriber"
unpack=true  # unpack .zip or not
chunk_size=8192  # chunk size in byte to download

audyn-download-song-describer \
root="${data_root}" \
song_describer_root="${song_describer_root}" \
unpack=${unpack} \
chunk_size=${chunk_size}
```
