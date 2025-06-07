import importlib
import os
import shutil
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ._hydra.utils import (
    instantiate,
    instantiate_cascade_text_to_wave,
    instantiate_criterion,
    instantiate_gan_discriminator,
    instantiate_gan_generator,
    instantiate_grad_clipper,
    instantiate_lr_scheduler,
    instantiate_metrics,
    instantiate_model,
    instantiate_optimizer,
)
from .cache import get_cache_dir, get_model_cache_dir
from .clip_grad import GANGradClipper, GradClipper
from .data import select_accelerator
from .data.dataloader import (
    DistributedDataLoader,
    DistributedDynamicBatchDataLoader,
    DistributedSequentialBatchDataLoader,
    DynamicBatchDataLoader,
    SequentialBatchDataLoader,
    WebLoaderWrapper,
)
from .data.dataset import (
    SortableTorchObjectDataset,
    TorchObjectDataset,
    WebDatasetWrapper,
    available_dump_formats,
)
from .distributed import is_distributed, setup_distributed
from .logging import get_logger

__all__ = [
    "audyn_cache_dir",
    "model_cache_dir",
    "clear_cache",
    "setup_config",
    "setup_system",
    "save_resolved_config",
    "set_seed",
    "convert_dataset_and_dataloader_to_ddp_if_possible",
    "convert_dataset_and_dataloader_format_if_necessary",
    "set_nodes_if_necessary",
    "set_compiler_if_necessary",
    "register_dump_format",
    "list_available_dump_formats",
    "is_available_dump_format",
    "instantiate",
    "instantiate_model",
    "instantiate_gan_generator",
    "instantiate_gan_discriminator",
    "instantiate_cascade_text_to_wave",
    "instantiate_optimizer",
    "instantiate_lr_scheduler",
    "instantiate_grad_clipper",
    "instantiate_criterion",
    "instantiate_metrics",
    "GradClipper",
    "GANGradClipper",
]

audyn_cache_dir = get_cache_dir()
model_cache_dir = get_model_cache_dir()


def clear_cache(**kwargs) -> None:
    """Remove cache directory by ``shutil.rmtree`` for Audyn package.

    Args:
        kwargs: Keyword arguments given to ``shutil.rmtree``.

    """
    shutil.rmtree(audyn_cache_dir, **kwargs)


def setup_system(config: DictConfig) -> None:
    """Set up config before training and evaluation.

    Args:
        config (DictConfig): Config to set up.

    .. note::

        This function is deprecated. Use ``audyn.utils.setup_config()`` instead.

    """
    warnings.warn(
        "audyn.utils.setup_system is deprecated. Use audyn.utils.setup_config instead.",
        UserWarning,
        stacklevel=2,
    )
    setup_config(config)


def setup_config(config: DictConfig) -> None:
    """Set up config before training and evaluation.

    Args:
        config (DictConfig): Config to set up.

    """
    if hasattr(config, "system"):
        full_config = config
        system_config = config.system
    else:
        warnings.warn(
            "System config is given to setup_config. Full configuration is recommended.",
            DeprecationWarning,
            stacklevel=2,
        )
        full_config = None
        system_config = config

    if full_config is not None:
        # for backward compatibility
        if full_config.preprocess.max_workers is None:
            cpu_count = os.cpu_count()

            if cpu_count is None:
                max_workers = 1
            else:
                max_workers = max(cpu_count // 2, 1)

            OmegaConf.update(
                full_config,
                "preprocess.max_workers",
                max_workers,
            )

    if full_config is not None:
        # for backward compatibility
        if not hasattr(full_config.train, "ddp_kwargs"):
            OmegaConf.update(
                full_config,
                "train.ddp_kwargs",
                None,
                force_add=True,
            )

        if not hasattr(full_config.test, "ddp_kwargs"):
            OmegaConf.update(
                full_config,
                "test.ddp_kwargs",
                None,
                force_add=True,
            )

    accelerator = select_accelerator(system_config)

    if accelerator in ["gpu", "cuda"]:
        from torch.backends import cudnn

        cudnn.benchmark = system_config.cudnn.benchmark
        cudnn.deterministic = system_config.cudnn.deterministic

    if full_config is not None:
        # overwrite full_config
        convert_dataset_and_dataloader_format_if_necessary(full_config)
        system_config = full_config.system

    if is_distributed(system_config):
        if full_config is None:
            warnings.warn(
                "System config is given to setup_config. In that case, "
                "training configuration is not converted to DDP.",
                UserWarning,
                stacklevel=2,
            )
        else:
            # overwrite full_config
            convert_dataset_and_dataloader_to_ddp_if_possible(full_config)
            system_config = full_config.system

        set_nodes_if_necessary(system_config)
        setup_distributed(system_config)

    set_compiler_if_necessary(system_config)

    if full_config is None:
        warnings.warn(
            "System config is given to setup_config. In that case, "
            "resolved configuration is not saved.",
            UserWarning,
            stacklevel=2,
        )
    else:
        save_resolved_config(full_config)

    set_seed(system_config.seed)


def save_resolved_config(
    config: Union[DictConfig, ListConfig], path: Optional[str] = None
) -> None:
    """Save resolved config.

    Args:
        config (DictConfig or ListConfig): Config to save.
        path (str, optional): Optional path to save config. If ``None``,
            ``os.path.join(output_dir, "resolved_config.yaml")`` is used, where ``output_dir``
            is defined by ``hydra``.

    """
    if path is None:
        # TODO: improve design
        try:
            output_dir = HydraConfig.get().runtime.output_dir
            path = os.path.join(output_dir, ".hydra", "resolved_config.yaml")
        except ValueError:
            return

    OmegaConf.save(config, path, resolve=True)


def set_seed(seed: int = 0) -> None:
    """Set random seeds."""
    import random

    # NOTE: random module is deprecated in Audyn.
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        # numpy is NOT a necessary package.
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def convert_dataset_and_dataloader_to_ddp_if_possible(config: DictConfig) -> None:
    """Convert dataset and data loader in config.train.dataloader.train and
    config.train.dataloader.validation for DDP.

    .. note::

        This function may overwrite config.train.dataloader.train
        and config.train.dataloader.validation.

    .. note::

        If conversion is required, you have to set environmental variables
        ``WORLD_SIZE`` and ``RANK``.

    """
    from .data.audioset.dataset import (
        DistributedPaSSTAudioSetWebDataset,
        DistributedWeightedAudioSetWebDataset,
        PaSSTAudioSetWebDataset,
        WeightedAudioSetWebDataset,
    )
    from .data.dnr.dataset import (
        DistributedRandomStemsDNRDataset,
        RandomStemsDNRDataset,
    )
    from .data.musdb18.dataset import (
        DistributedRandomStemsMUSDB18Dataset,
        RandomStemsMUSDB18Dataset,
    )

    subset_names = ["train", "validation"]

    dataset_configs = {}
    dataloader_configs = {}

    # resolve configs at first
    for subset in subset_names:
        dataset_config = getattr(config.train.dataset, subset)
        dataset_config = OmegaConf.to_container(dataset_config, resolve=True)
        dataset_configs[subset] = OmegaConf.create(dataset_config)

        dataloader_config = getattr(config.train.dataloader, subset)
        dataloader_config = OmegaConf.to_container(dataloader_config, resolve=True)
        dataloader_configs[subset] = OmegaConf.create(dataloader_config)

    for subset in subset_names:
        dataset_config = dataset_configs[subset]
        dataloader_config = dataloader_configs[subset]

        # split _target_ into names of package, module, variable
        # e.g.
        #         _target_: audyn.utils.data.dataset.WebDatasetWrapper.instantiate_dataset
        #     package_name: audyn
        #         mod_name: audyn.utils.data.dataset
        #         var_name: WebDatasetWrapper
        #         fn_name: instantiate_dataset
        #
        #         _target_: audyn.utils.data.audioset.dataset.DistributedWeightedAudioSetWebDataset
        #     package_name: audyn
        #         mod_name: audyn.utils.data.audioset.dataset
        #         var_name: DistributedWeightedAudioSetWebDataset
        #          fn_name: None
        dataset_mod_name, dataset_var_name = dataset_config._target_.rsplit(".", maxsplit=1)

        try:
            dataset_fn_name = None
            imported_dataset_module = importlib.import_module(dataset_mod_name)
            dataset_factory_fn = getattr(imported_dataset_module, dataset_var_name)
        except ModuleNotFoundError:
            dataset_fn_name = dataset_var_name
            dataset_mod_name, dataset_var_name = dataset_mod_name.rsplit(".", maxsplit=1)
            imported_dataset_module = importlib.import_module(dataset_mod_name)
            dataset_cls = getattr(imported_dataset_module, dataset_var_name)
            dataset_factory_fn = getattr(dataset_cls, dataset_fn_name)

        # split _target_ into names of package, module, variable
        # e.g.
        #     _target_: audyn.utils.data.SequentialBatchDataLoader
        # package_name: audyn
        #     mod_name: audyn.utils.data
        #     var_name: SequentialBatchDataLoader
        dataloader_mod_name, dataloader_var_name = dataloader_config._target_.rsplit(
            ".", maxsplit=1
        )
        dataloader_package_name, *_ = dataloader_mod_name.split(".", maxsplit=1)

        try:
            dataloader_fn_name = None
            imported_dataloader_module = importlib.import_module(dataloader_mod_name)
            dataloader_factory_fn = getattr(imported_dataloader_module, dataloader_var_name)
        except ModuleNotFoundError:
            dataloader_fn_name = dataloader_var_name
            dataloader_mod_name, dataloader_var_name = dataloader_mod_name.rsplit(".", maxsplit=1)
            imported_dataloader_module = importlib.import_module(dataloader_mod_name)
            dataloader_cls = getattr(imported_dataloader_module, dataloader_var_name)
            dataloader_factory_fn = getattr(dataloader_cls, dataloader_fn_name)

        if dataloader_package_name == "torch":
            if dataloader_factory_fn is DataLoader:
                # may be converted to distributed data loader
                if "sampler" in dataloader_config.keys():
                    sampler = dataloader_config.sampler

                    if sampler is not None and sampler._target_ is not None:
                        sampler_target = sampler._target_
                        sampler_mod_name, sampler_var_name = sampler_target.rsplit(".", maxsplit=1)
                        sampler_cls = getattr(
                            importlib.import_module(sampler_mod_name), sampler_var_name
                        )

                        if sampler_cls is DistributedSampler:
                            # torch.utils.data.DistributedSampler is expected to work well.
                            pass
                        else:
                            raise ValueError(
                                f"Sampler {sampler_target} cannot be automatically converted "
                                "to DDP-supported."
                            )

                dataset_config = OmegaConf.to_container(dataset_config)
                dataloader_config = OmegaConf.to_container(dataloader_config)

                if "seed" in dataloader_config.keys():
                    # NOTE: Since torch.utils.data.DataLoader does not support seed, seed
                    #       should not be defined in config.train.dataloader.train for proper use.
                    seed = dataloader_config["seed"]
                else:
                    seed = "${system.seed}"

                if dataset_factory_fn == WebDatasetWrapper.instantiate_dataset:
                    # DistributedDataLoader is not available
                    # wds.split_by_node plays such role
                    pass
                elif (
                    dataset_factory_fn == WeightedAudioSetWebDataset.instantiate_dataset
                    or dataset_factory_fn == PaSSTAudioSetWebDataset.instantiate_dataset
                ):
                    if dataset_factory_fn == WeightedAudioSetWebDataset.instantiate_dataset:
                        # WeightedAudioSetWebDataset.instantiate_dataset
                        # -> DistributedWeightedAudioSetWebDataset.instantiate_dataset
                        ddp_dataset_factory_fn = DistributedWeightedAudioSetWebDataset
                    elif dataset_factory_fn == PaSSTAudioSetWebDataset.instantiate_dataset:
                        # PaSSTAudioSetWebDataset.instantiate_dataset
                        # -> DistributedPaSSTAudioSetWebDataset.instantiate_dataset
                        ddp_dataset_factory_fn = DistributedPaSSTAudioSetWebDataset
                    else:
                        raise ValueError(
                            f"Invalid dataset_factory_fn {dataset_factory_fn} is found."
                        )

                    ddp_target = ".".join(
                        [
                            ddp_dataset_factory_fn.__module__,
                            ddp_dataset_factory_fn.__name__,
                            ddp_dataset_factory_fn.instantiate_dataset.__name__,
                        ]
                    )
                    additional_ddp_config = {
                        "_target_": ddp_target,
                        "num_replicas": int(os.environ["WORLD_SIZE"]),
                        "rank": int(os.environ["RANK"]),
                        "seed": seed,
                    }
                    dataset_config.update(additional_ddp_config)
                    OmegaConf.update(
                        config, f"train.dataset.{subset}", dataset_config, merge=False
                    )
                elif (
                    dataset_factory_fn is WeightedAudioSetWebDataset
                    or dataset_factory_fn is PaSSTAudioSetWebDataset
                ):
                    if dataset_factory_fn is WeightedAudioSetWebDataset:
                        # WeightedAudioSetWebDataset -> DistributedWeightedAudioSetWebDataset
                        ddp_dataset_factory_fn = DistributedWeightedAudioSetWebDataset
                    elif dataset_factory_fn is PaSSTAudioSetWebDataset:
                        # PaSSTAudioSetWebDataset -> DistributedPaSSTAudioSetWebDataset
                        ddp_dataset_factory_fn = DistributedPaSSTAudioSetWebDataset
                    else:
                        raise ValueError(
                            f"Invalid dataset_factory_fn {dataset_factory_fn} is found."
                        )

                    ddp_target = ".".join(
                        [
                            ddp_dataset_factory_fn.__module__,
                            ddp_dataset_factory_fn.__name__,
                        ]
                    )
                    additional_ddp_config = {
                        "_target_": ddp_target,
                        "num_replicas": int(os.environ["WORLD_SIZE"]),
                        "rank": int(os.environ["RANK"]),
                        "seed": seed,
                    }
                    dataset_config.update(additional_ddp_config)
                    OmegaConf.update(
                        config, f"train.dataset.{subset}", dataset_config, merge=False
                    )
                elif dataset_factory_fn == RandomStemsMUSDB18Dataset:
                    ddp_dataset_factory_fn = DistributedRandomStemsMUSDB18Dataset

                    ddp_target = ".".join(
                        [
                            ddp_dataset_factory_fn.__module__,
                            ddp_dataset_factory_fn.__name__,
                        ]
                    )
                    additional_ddp_config = {
                        "_target_": ddp_target,
                        "num_replicas": int(os.environ["WORLD_SIZE"]),
                        "rank": int(os.environ["RANK"]),
                        "seed": seed,
                    }
                    dataset_config.update(additional_ddp_config)
                    OmegaConf.update(
                        config, f"train.dataset.{subset}", dataset_config, merge=False
                    )
                elif dataset_factory_fn == RandomStemsDNRDataset:
                    ddp_dataset_factory_fn = DistributedRandomStemsDNRDataset

                    ddp_target = ".".join(
                        [
                            ddp_dataset_factory_fn.__module__,
                            ddp_dataset_factory_fn.__name__,
                        ]
                    )
                    additional_ddp_config = {
                        "_target_": ddp_target,
                        "num_replicas": int(os.environ["WORLD_SIZE"]),
                        "rank": int(os.environ["RANK"]),
                        "seed": seed,
                    }
                    dataset_config.update(additional_ddp_config)
                    OmegaConf.update(
                        config, f"train.dataset.{subset}", dataset_config, merge=False
                    )
                else:
                    # DataLoader -> DistributedDataLoader
                    ddp_target = ".".join(
                        [DistributedDataLoader.__module__, DistributedDataLoader.__name__]
                    )
                    additional_ddp_config = {
                        "_target_": ddp_target,
                        "num_replicas": int(os.environ["WORLD_SIZE"]),
                        "rank": int(os.environ["RANK"]),
                        "seed": seed,
                    }
                    dataloader_config.update(additional_ddp_config)
                    OmegaConf.update(
                        config, f"train.dataloader.{subset}", dataloader_config, merge=False
                    )
            else:
                _warn_unexpected_dataloader_for_ddp(dataloader_factory_fn)
        elif dataloader_package_name == "audyn":
            if (
                dataloader_factory_fn is SequentialBatchDataLoader
                or dataloader_factory_fn is DynamicBatchDataLoader
            ):
                dataloader_config = OmegaConf.to_container(dataloader_config)

                if "seed" in dataloader_config.keys():
                    seed = dataloader_config["seed"]
                else:
                    seed = "${system.seed}"

                # should be converted to distributed data loader
                # SequentialBatchDataLoader -> DistributedSequentialBatchDataLoader
                # DynamicBatchDataLoader -> DistributedDynamicBatchDataLoader
                ddp_target = ".".join(
                    [dataloader_mod_name, "Distributed" + dataloader_factory_fn.__name__]
                )
                additional_ddp_config = {
                    "_target_": ddp_target,
                    "num_replicas": int(os.environ["WORLD_SIZE"]),
                    "rank": int(os.environ["RANK"]),
                    "seed": seed,
                }
                dataloader_config.update(additional_ddp_config)
                OmegaConf.update(
                    config, f"train.dataloader.{subset}", dataloader_config, merge=False
                )
            elif (
                dataloader_factory_fn is DistributedDataLoader
                or dataloader_factory_fn is DistributedSequentialBatchDataLoader
                or dataloader_factory_fn is DistributedDynamicBatchDataLoader
            ):
                # These data loaders support DDP.
                pass
            elif dataloader_factory_fn == WebLoaderWrapper.instantiate_dataloader:
                pass
            else:
                _warn_unexpected_dataloader_for_ddp(dataloader_factory_fn)
        else:
            _warn_unexpected_dataloader_for_ddp(dataloader_factory_fn)


def convert_dataset_and_dataloader_format_if_necessary(config: DictConfig) -> None:
    """Convert dataset and data loader in config.train for torch or WebDataset if necessary.

    The data format should be specified by config.preprocess.dump_format.

    .. note::

        This function may overwrite config.train.dataset and config.train.dataloader.

    """
    from .data.audioset.dataset import (
        DistributedPaSSTAudioSetWebDataset,
        DistributedWeightedAudioSetWebDataset,
        PaSSTAudioSetWebDataset,
        WeightedAudioSetWebDataset,
    )

    logger = get_logger(convert_dataset_and_dataloader_format_if_necessary.__name__)

    dump_format = config.preprocess.dump_format
    dataset_config = config.train.dataset
    dataloader_config = config.train.dataloader

    if dump_format == "webdataset":
        _, train_dataset_cls, _ = _load_package_class_function_from_config(dataset_config.train)
        _, validation_dataset_cls, _ = _load_package_class_function_from_config(
            dataset_config.validation
        )

        if (
            train_dataset_cls is WeightedAudioSetWebDataset
            or train_dataset_cls is PaSSTAudioSetWebDataset
            or train_dataset_cls is DistributedWeightedAudioSetWebDataset
            or train_dataset_cls is DistributedPaSSTAudioSetWebDataset
        ):
            train_dataset_kwargs = OmegaConf.to_container(dataset_config.train)
            train_dataset_target = train_dataset_kwargs.pop("_target_")
            train_dataloader_kwargs = OmegaConf.to_container(dataloader_config.train)
            train_dataloader_target = train_dataloader_kwargs.pop("_target_")

            if "shuffle" in train_dataloader_kwargs and train_dataloader_kwargs["shuffle"]:
                train_dataloader_kwargs["shuffle"] = False
                logger.info(
                    "shuffle=True is replaced with shuffle=False in config of data loader."
                )
        else:
            train_dataset_target, train_dataset_kwargs = _search_webdataset_format_dataset(
                dataset_config.train
            )
            (
                train_dataloader_target,
                train_dataloader_kwargs,
            ) = _search_webdataset_format_dataloader(dataloader_config.train)

        if (
            validation_dataset_cls is WeightedAudioSetWebDataset
            or validation_dataset_cls is PaSSTAudioSetWebDataset
            or validation_dataset_cls is DistributedWeightedAudioSetWebDataset
            or validation_dataset_cls is DistributedPaSSTAudioSetWebDataset
        ):
            validation_dataset_kwargs = OmegaConf.to_container(dataset_config.validation)
            validation_dataset_target = validation_dataset_kwargs.pop("_target_")
            validation_dataloader_kwargs = OmegaConf.to_container(dataloader_config.validation)
            validation_dataloader_target = validation_dataloader_kwargs.pop("_target_")

            if (
                "shuffle" in validation_dataloader_kwargs
                and validation_dataloader_kwargs["shuffle"]
            ):
                validation_dataloader_kwargs["shuffle"] = False
                logger.info(
                    "shuffle=True is replaced with shuffle=False in config of data loader."
                )
        else:
            validation_dataset_target, validation_dataset_kwargs = (
                _search_webdataset_format_dataset(dataset_config.validation)
            )
            (
                validation_dataloader_target,
                validation_dataloader_kwargs,
            ) = _search_webdataset_format_dataloader(dataloader_config.validation)

        OmegaConf.update(
            config,
            "train.dataset.train",
            {"_target_": train_dataset_target, **train_dataset_kwargs},
            merge=False,
        )
        OmegaConf.update(
            config,
            "train.dataset.validation",
            {"_target_": validation_dataset_target, **validation_dataset_kwargs},
            merge=False,
        )
        OmegaConf.update(
            config,
            "train.dataloader.train",
            {"_target_": train_dataloader_target, **train_dataloader_kwargs},
            merge=False,
        )
        OmegaConf.update(
            config,
            "train.dataloader.validation",
            {"_target_": validation_dataloader_target, **validation_dataloader_kwargs},
            merge=False,
        )
    elif dump_format in available_dump_formats:
        # TODO: format conversion other than webdataset.
        pass
    else:
        raise ValueError(f"Unknown dump format {dump_format} is detected.")


def set_nodes_if_necessary(config: DictConfig) -> None:
    """Set config.distributed.nodes if necessary.

    .. note::

        This function may overwrite config.distributed.nodes.

    """
    if config.distributed.nodes is None:
        OmegaConf.update(
            config,
            "distributed.nodes",
            1,
        )


def set_compiler_if_necessary(config: DictConfig) -> None:
    """Set config.compile if necessary.

    .. note::

        This function may overwrite config.compile.

    """
    from ._torch.compile import is_gpu_supported, is_supported

    if config.compile.enable is None:
        OmegaConf.update(
            config,
            "compile.enable",
            False,
        )
    else:
        if config.compile.enable and not is_supported():
            warnings.warn(
                "Since torch.compile is not available on your system, "
                "we overwrite config.compile.enable=False.",
                UserWarning,
                stacklevel=2,
            )
            OmegaConf.update(
                config,
                "compile.enable",
                False,
            )

        if config.compile.enable and config.accelerator in ["gpu", "cuda"]:
            if not is_gpu_supported():
                warnings.warn(
                    "Since torch.compile is not available on your device, "
                    "we overwrite config.compile.enable=False.",
                    UserWarning,
                    stacklevel=2,
                )
                OmegaConf.update(
                    config,
                    "compile.enable",
                    False,
                )


def register_dump_format(dump_format: str) -> list[str]:
    """Register dump format to available_dump_formats.

    Args:
        dump_format (str): Dump format to register.

    Returns:
        list: List of available dump formats.

    Examples:

        >>> from audyn.utils import list_available_dump_formats, register_dump_format
        >>> list_available_dump_formats()
        ['torch', 'webdataset', 'birdclef2024', 'musdb18', 'dnr-v2', 'custom']
        >>> register_dump_format("new-format")
        ['torch', 'webdataset', 'birdclef2024', 'musdb18', 'dnr-v2', 'custom', 'new-format']
        >>> list_available_dump_formats()
        ['torch', 'webdataset', 'birdclef2024', 'musdb18', 'dnr-v2', 'custom', 'new-format']

    """
    if dump_format in available_dump_formats:
        pass
    else:
        available_dump_formats.append(dump_format)

    return available_dump_formats


def list_available_dump_formats() -> list[str]:
    """List available dump formats.

    Returns:
        list: List of available dump formats.

    Examples:

        >>> from audyn.utils import list_available_dump_formats
        >>> list_available_dump_formats()
        ['torch', 'webdataset', 'birdclef2024', 'musdb18', 'dnr-v2', 'custom']

    """
    return available_dump_formats


def is_available_dump_format(dump_format: str) -> bool:
    """Check if dump format is included in available_dump_formats.

    Args:
        dump_format (str): Dump format to check.

    Returns:
        bool: If ``True``, dump format is available.

    Examples:

        >>> from audyn.utils import is_available_dump_format, register_dump_format
        >>> is_available_dump_format("new-format")
        False
        >>> register_dump_format("new-format")
        ['torch', 'webdataset', 'birdclef2024', 'musdb18', 'dnr-v2', 'custom', 'new-format']
        >>> is_available_dump_format("new-format")
        True

    """
    if dump_format in available_dump_formats:
        return True
    else:
        return False


def _search_webdataset_format_dataset(config: DictConfig) -> Tuple[str, Dict[str, Any]]:
    from .data.audioset.dataset import (
        DistributedPaSSTAudioSetWebDataset,
        DistributedWeightedAudioSetWebDataset,
        PaSSTAudioSetWebDataset,
        WeightedAudioSetWebDataset,
    )

    # split _target_ into names of package, module, variable
    # e.g.
    #     _target_: audyn.utils.data.TorchObjectDataset
    # package_name: audyn
    #     mod_name: audyn.utils.data
    #     var_name: TorchObjectDataset
    #
    #     _target_: audyn.utils.data.dataset.WebDatasetWrapper.instantiate_dataset
    # package_name: audyn
    #     mod_name: audyn.utils.data.dataset
    #     var_name: WebDatasetWrapper
    #      fn_name: instantiate_dataset
    pkg, cls, fn = _load_package_class_function_from_config(config)
    config = OmegaConf.to_container(config)
    target = config.pop("_target_")

    if pkg.__name__ == "torch":
        assert fn is None

        _warn_unexpected_dataset_for_webdataset(cls)
    elif pkg.__name__ == "audyn":
        # NOTE: WebDatasetWrapper.instantiate_dataset is not a class.
        if (
            cls is WebDatasetWrapper
            or cls is WeightedAudioSetWebDataset
            or cls is DistributedWeightedAudioSetWebDataset
            or cls is PaSSTAudioSetWebDataset
            or cls is DistributedPaSSTAudioSetWebDataset
        ):
            # WebDataset is supported by WebDatasetWrapper.
            pass
        elif cls is TorchObjectDataset:
            # TorchObjectDataset is convertible to WebDatasetWrapper.
            target = "audyn.utils.data.WebDatasetWrapper.instantiate_dataset"
        elif cls is SortableTorchObjectDataset:
            raise NotImplementedError("SortableTorchObjectDataset cannot be converted")
        else:
            _warn_unexpected_dataset_for_webdataset(cls)
    else:
        _warn_unexpected_dataset_for_webdataset(cls)

    return target, config


def _search_webdataset_format_dataloader(config: DictConfig) -> Tuple[str, Dict[str, Any]]:
    logger = get_logger(_search_webdataset_format_dataloader.__name__)

    # split _target_ into names of package, module, variable
    # e.g.
    #     _target_: audyn.utils.data.SequentialBatchDataLoader
    # package_name: audyn
    #     mod_name: audyn.utils.data
    #     var_name: SequentialBatchDataLoader
    pkg, cls, _ = _load_package_class_function_from_config(config)
    config = OmegaConf.to_container(config)
    target = config.pop("_target_")

    if pkg.__name__ == "torch":
        if cls is DataLoader:
            # WebDataset is supported by DataLoader.
            if "shuffle" in config and config["shuffle"]:
                config["shuffle"] = False
                logger.info(
                    "shuffle=True is replaced with shuffle=False in config of data loader."
                )
            target = "audyn.utils.data.WebLoaderWrapper.instantiate_dataloader"
        else:
            warnings.warn(
                f"{cls.__name__} cannot be converted for WebDataset.",
                UserWarning,
                stacklevel=2,
            )
    elif pkg.__name__ == "audyn":
        if (
            cls is DistributedDataLoader
            or cls is SequentialBatchDataLoader
            or cls is DistributedSequentialBatchDataLoader
            or cls is DynamicBatchDataLoader
            or cls is DistributedDynamicBatchDataLoader
        ):
            raise NotImplementedError(
                f"Automatic conversion of {cls.__name__} to WebDataset is not supported now."
            )
        else:
            warnings.warn(
                f"{cls.__name__} cannot be converted for WebDataset.",
                UserWarning,
                stacklevel=2,
            )
    else:
        _warn_unexpected_dataloader_for_webdataset(cls)

    return target, config


def _load_package_class_function_from_config(config: DictConfig) -> Tuple[Any, Any, Any]:
    config = OmegaConf.to_container(config)
    target = config.pop("_target_")
    mod_name, var_name = target.rsplit(".", maxsplit=1)
    package_name, *_ = mod_name.split(".", maxsplit=1)
    pkg = importlib.import_module(package_name)

    try:
        fn_name = None
        imported_module = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        fn_name = var_name
        mod_name, var_name = mod_name.rsplit(".", maxsplit=1)
        imported_module = importlib.import_module(mod_name)

    cls = getattr(imported_module, var_name)

    if fn_name is None:
        fn = None
    else:
        fn = getattr(cls, fn_name)

    return pkg, cls, fn


def _warn_unexpected_dataset_for_webdataset(module: Any) -> None:
    """Warn unexpected dataset that cannot be converted to WebDataset-supported one."""
    warnings.warn(
        f"Dataset {module.__name__} cannot be converted for WebDataset.",
        UserWarning,
        stacklevel=2,
    )


def _warn_unexpected_dataloader_for_webdataset(module: Any) -> None:
    """Warn unexpected dataloader that cannot be converted to WebDataset-supported one."""
    warnings.warn(
        f"Dataloader {module.__name__} cannot be converted for WebDataset.",
        UserWarning,
        stacklevel=2,
    )


def _warn_unexpected_dataloader_for_ddp(module: Any) -> None:
    """Warn unexpected data loader that cannot be converted to DDP-supported one.

    Both torch's and audyn's data loaders are supported.
    """
    warnings.warn(
        f"Unexpected data loader {module.__name__} is found. "
        "There is not rule to convert to DDP-supported loader.",
        UserWarning,
        stacklevel=2,
    )
