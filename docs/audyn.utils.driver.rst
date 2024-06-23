audyn.utils.driver
==================

``audyn.utils.driver`` provides drivers to train and evaluate models.


Classes
-------

.. autoclass:: audyn.utils.driver.AutoTrainer
    :members: build_from_config

.. autoclass:: audyn.utils.driver.BaseTrainer
    :members: build_from_config, run, train_one_epoch, validate_one_epoch, infer_one_batch,
        train_one_iteration, validate_one_iteration, infer_one_iteration,
        unscale_optimizer_if_necessary, clip_gradient_if_necessary, optimizer_step, lr_scheduler_step, display_loss, load_checkpoint, save_checkpoint_if_necessary,
        write_train_duration_if_necessary, write_train_spectrogram_if_necessary, write_train_waveform_if_necessary, write_train_audio_if_necessary, write_train_image_if_necessary,
        write_validation_duration_if_necessary, write_validation_spectrogram_if_necessary, write_validation_waveform_if_necessary, write_validation_audio_if_necessary, write_validation_image_if_necessary,
        write_inference_duration_if_necessary, write_inference_spectrogram_if_necessary, write_inference_waveform_if_necessary, write_inference_audio_if_necessary, write_inference_image_if_necessary,
        set_epoch_if_necessary, set_commit_hash

.. autoclass:: audyn.utils.driver.BaseGenerator
    :members: build_from_config, run, load_checkpoint,
        save_inference_torch_dump_if_necessary, save_inference_audio_if_necessary, save_inference_spectrogram_if_necessary,
        remove_weight_norm_if_necessary

.. autoclass:: audyn.utils.driver.TextToFeatTrainer

.. autoclass:: audyn.utils.driver.FeatToWaveTrainer

.. autoclass:: audyn.utils.driver.TextToWaveTrainer

.. autoclass:: audyn.utils.driver.FeatToWaveGenerator

.. autoclass:: audyn.utils.driver.CascadeTextToWaveGenerator

.. autoclass:: audyn.utils.driver.GANTrainer
    :members: build_from_config, run,
        train_one_epoch, validate_one_epoch, infer_one_batch,
        count_num_parameters, display_model, display_loss, unscale_optimizer_if_necessary, clip_gradient_if_necessary,
        load_checkpoint, save_checkpoint

.. autoclass:: audyn.utils.driver.GANGenerator
    :members: load_checkpoint, run
