audyn.utils.data.musdb18
========================

``audyn.utils.data.musdb18`` provides utilities for MUSDB18 dataset.

Example
-------

.. code-block::

    >>> from audyn.utils.data.musdb18 import MUSDB18
    >>> root = "./MUSDB18"
    >>> subset = "train"
    >>> dataset = MUSDB18(root=root, subset=subset)
    >>> track = dataset[0]
    >>> waveform, sample_rate = track.mixture
    >>> waveform.size()
    torch.Size([2, 7552000])
    >>> sample_rate
    44100
    >>> # drums, bass, other, vocals
    >>> waveform, sample_rate = track.drums
    >>> waveform.size()
    torch.Size([2, 7552000])
    >>> sample_rate
    44100
    >>> # stems
    >>> waveform, sample_rate = track.stems
    >>> waveform.size()
    torch.Size([5, 2, 7552000])
    >>> sample_rate
    44100
    >>> # set frame_offset and num_frames
    >>> chunk_start = 10.2
    >>> chunk_duration = 5.5
    >>> track.frame_offset = int(chunk_start * sample_rate)
    >>> track.num_frames = int(chunk_duration * sample_rate)
    >>> waveform, sample_rate = track.drums
    >>> waveform.size()
    torch.Size([2, 242550])
    >>> sample_rate
    44100

Classes
-------

Dataset
^^^^^^^

.. autoclass:: audyn.utils.data.musdb18.MUSDB18
