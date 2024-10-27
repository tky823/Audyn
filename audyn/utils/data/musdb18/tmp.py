class DistributedRandomStemsMUSDB18Dataset(RandomStemsMUSDB18Dataset):
    """MUSDB18 dataset for random mixing using distributed data parallel strategy.

    Args:
        root (str): Root of MUSDB18 dataset.
        subset (str): ``train``, ``validation``, or ``test``.
        duration (float): Duration of waveform slice.
        drums_key (str): Key to store ``drums`` waveform.
        bass_key (str): Key to store ``bass`` waveform.
        other_key (str): Key to store ``other`` waveform.
        vocals_key (str): Key to store ``vocals`` waveform.
        sample_rate_key (str): Key to store sampling rate.
        filename_key (str): Key to store filename.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        seed (int): Random seed to set sampler state.

    .. note::

        We assume following structure.

        .. code-block:: shell

            - root/  # typically MUSDB18, MUSDB18-HQ, MUSDB18-7s
                |- train/
                    |- A Classic Education - NightOwl/
                        |- mixture.wav
                        |- drums.wav
                        |- bass.wav
                        |- other.wav
                        |- vocals.wav
                    ...
                |- test/
                    ...

    """

    def __init__(
        self,
        root: str,
        subset: str,
        duration: float,
        drums_key: str = "drums",
        bass_key: str = "bass",
        other_key: str = "other",
        vocals_key: str = "vocals",
        sample_rate_key: str = "sample_rate",
        filename_key: str = "filename",
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            root,
            subset,
            duration,
            drums_key=drums_key,
            bass_key=bass_key,
            other_key=other_key,
            vocals_key=vocals_key,
            sample_rate_key=sample_rate_key,
            filename_key=filename_key,
            seed=seed,
        )

        self.sampler = DistributedRandomStemsMUSDB18Sampler(
            self.track_names,
            num_replicas=num_replicas,
            rank=rank,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        from . import sources

        root = self.root
        source_keys = self.source_keys
        sample_rate_key = self.sample_rate_key
        filename_key = self.filename_key

        subset_dir = "test" if self.subset == "test" else "train"

        if self.worker_id is None:
            # should be initialized
            worker_info = get_worker_info()

            if worker_info is None:
                self.worker_id = 0
                num_workers = 1
            else:
                self.worker_id = worker_info.id
                num_workers = worker_info.num_workers

            # set generator state
            seed = self.seed + self.sampler.rank * num_workers + self.worker_id
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

            # set sampler state
            sampler = self.sampler
            num_total_samples = sampler.num_samples_per_source
            num_samples_per_worker = num_total_samples // num_workers

            if self.worker_id < num_total_samples % num_workers:
                num_samples_per_worker += 1

            self.sampler = DistributedRandomStemsMUSDB18Sampler(
                sampler.track_names,
                replacement=sampler.replacement,
                num_samples=num_samples_per_worker,
                num_replicas=sampler.num_replicas,
                rank=sampler.rank,
                drop_last=sampler.drop_last,
                generator=sampler.generator,
            )

        for indices in self.sampler:
            track_names = []
            feature = {}

            assert len(indices) == len(sources) == len(source_keys)

            for idx, source, source_key in zip(indices, sources, source_keys):
                track_name = self.track_names[idx]
                track_names.append(track_name)
                filename = f"{track_name}/{source}.wav"
                path = os.path.join(root, subset_dir, filename)
                waveform, sample_rate = self.load_sliced_audio(path)

                if sample_rate_key in feature:
                    assert feature[sample_rate_key].item() == sample_rate
                else:
                    feature[sample_rate_key] = torch.tensor(sample_rate, dtype=torch.long)

                feature[source_key] = waveform

            feature[filename_key] = "+".join(track_names)

            yield feature
