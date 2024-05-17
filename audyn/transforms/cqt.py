import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as aT

from ..utils.music import note_to_hz

__all__ = [
    "ConstantQTransform",
    "CQT",
]

_invalid_domain_message = "Invalid domain {domain} is given."
_invalid_f_max_message = (
    "Max frequency ({f_max}) should not exceed Nyquist frequency ({f_nyq})."
    " Set smaller f_max or n_bins."
)


class ConstantQTransform(nn.Module):
    """Constant-Q transform.

    Args:
        sample_rate (int): Sampling rate.
        hop_length (int, optional): Hop length of CQT. If ``None``, 1/4 of minimum filter length
            at highest analysis frequency is used.
        f_min (float): Minimum frequency. Default: ``audyn.utils.music.note_to_hz("C1")``.
        f_max (float, optional): Maximum frequency.
        pad (float): Value to pad. Default: ``0``.
        n_bins (int): Number of bins. Default: ``84``.
        bins_per_octave (float, optional): Number of bins per octave.
        window_fn (callable, optional): Default: ``torch.hann_window``.
        wkwargs (dict, optional): Keyword arguments given to ``window_fn``.
        center (bool): Whether to pad waveform on both sides.
        pad_mode (str): Padding mode used for ``center`` is ``True``. Default: ``reflect``.
        scaling_factor (float): Scaling factor of CQT. Default: ``1``.
        by_octave (bool): If ``True``, CQT algorithm in 2010 is used. Default: ``True``.
        domain (str, optional): Domain to compute CQT. ``time`` and ``freq`` are supported.
            Default value differs by ``by_octave``. If ``by_octave=True, ``domain``
            defaults to ``"freq"``. , ``domain`` defaults to ``"time"``.
        sparse (bool, optional): If ``True``, sparse algorithm is used. This option is supported
            only when ``by_octave=True`` and ``domain="freq"``. Default value differs
            by ``domain``. If ``domain="time", ``sparse`` defaults to ``False``.
            Otherwise (``domain="freq"), ``sparse`` defaults to ``True``.
        threshold (float, optional): Threshold is used when following conditions are satisfied:
            - ``by_octave=True``
            - ``domain="freq"``
            - ``sparse=True``
        kwargs: Keyword arguments. When ``by_octave=True``, ``downsample`` can be specified.

    Examples:

        >>> import torch
        >>> import torchaudio
        >>> import torchaudio.transforms as aT
        >>> from audyn.utils.music import note_to_hz
        >>> from audyn.transforms.cqt import ConstantQTransform
        >>> path = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
        >>> waveform, sample_rate = torchaudio.load(path)
        >>> waveform.size()
        torch.Size([1, 54400])
        >>> sample_rate
        16000
        >>> f_nyq = sample_rate / 2
        >>> f_nyq
        8000.0
        >>> f_min, hop_length, n_bins, bins_per_octave = note_to_hz("C1"), 512, 48, 12  # parameters of CQT
        >>> f_min * (2 ** (n_bins / bins_per_octave))  # f_max
        523.2511306011972  # f_max, which is lower than f_nyq by a large margin.
        >>> transform = ConstantQTransform(sample_rate, hop_length=hop_length, f_min=f_min, n_bins=n_bins, bins_per_octave=bins_per_octave)
        >>> spectrogram = transform(waveform)
        >>> spectrogram.size()
        torch.Size([1, 48, 107])
        >>> # for computational efficiency
        >>> new_sample_rate = 4000  # f_max is less than new_sample_rate / 2
        >>> new_hop_length = hop_length // (sample_rate // new_sample_rate)
        >>> new_hop_length
        128
        >>> resampler = aT.Resample(sample_rate, new_sample_rate)
        >>> new_waveform = resampler(waveform)
        >>> new_waveform.size()
        torch.Size([1, 13600])
        >>> new_transform = ConstantQTransform(new_sample_rate, hop_length=new_hop_length, f_min=f_min, n_bins=n_bins, bins_per_octave=bins_per_octave)
        >>> new_spectrogram = new_transform(new_waveform)
        >>> new_spectrogram.size()
        torch.Size([1, 48, 107])
        >>> torch.mean(torch.abs(new_spectrogram - spectrogram))
        tensor(0.0002)

    .. note::

        When the maximum center frequency is significantly lower than the Nyquist frequency,
        it is highly recommended to downsample a waveform before CQT for computational efficiency
        as shown in the example above.

    """  # noqa: E501

    def __init__(
        self,
        sample_rate: int,
        hop_length: Optional[int] = None,
        f_min: float = note_to_hz("C1"),
        f_max: Optional[float] = None,
        pad: float = 0,
        n_bins: int = 84,
        bins_per_octave: Optional[float] = None,
        window_fn: Optional[Callable[[int], torch.Tensor]] = torch.hann_window,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        scaling_factor: float = 1,
        by_octave: bool = True,
        domain: Optional[str] = None,
        sparse: Optional[bool] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        f_max, bins_per_octave = _set_f_max_and_bins_per_octave(
            f_min,
            f_max=f_max,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )

        if f_max > sample_rate / 2:
            raise ValueError(_invalid_f_max_message.format(f_max=f_max, f_nyq=sample_rate / 2))

        if domain is None:
            if by_octave:
                domain = "freq"
            else:
                domain = "time"

        if domain == "time":
            if sparse is None:
                sparse = False

            assert not sparse, "sparse=True is not supported when domain='time'."
        elif domain == "freq":
            if not by_octave:
                raise ValueError("When domain='freq', only by_octave=True is supported.")

            if sparse is None:
                sparse = True
        else:
            raise ValueError(_invalid_domain_message.format(domain=domain))

        if by_octave:
            valid_kwargs = {"downsample"}
            invalid_kwargs = set(kwargs.keys()) - valid_kwargs

            assert invalid_kwargs == set(), "Invalid keyword arguments {} are given".format(
                invalid_kwargs
            )
        else:
            assert (
                not sparse
            ), "Sparse implementation is not supported when by_octave=False. Set sparse=False."
            assert len(kwargs) == 0, "Invalid keyword arguments {} are given".format(
                set(kwargs.keys())
            )

        kernel = build_temporal_kernel(
            sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_bins=n_bins,
            window_fn=window_fn,
            wkwargs=wkwargs,
            scaling_factor=scaling_factor,
        )

        if by_octave:
            assert isinstance(
                bins_per_octave, int
            ), "bins_per_octave should be int when by_octave=True."

            num_repeats = math.ceil(n_bins / bins_per_octave)
            divided_by = 2 ** (num_repeats - 1)

            if hop_length is None:
                min_filter_length = compute_filter_length(
                    f_max,
                    sample_rate,
                    bins_per_octave=bins_per_octave,
                    scaling_factor=scaling_factor,
                )
                hop_length = min_filter_length // 4

                padding = (divided_by - hop_length % divided_by) % divided_by
                hop_length = hop_length + padding

            assert (
                hop_length % divided_by == 0
            ), f"Given hop length ({hop_length}) is not divisible by {divided_by}."

            # to reduce error, extract one octave kernel from full-band
            _, kernel = torch.split(kernel, [n_bins - bins_per_octave, bins_per_octave], dim=0)

            # remove zero padding
            f_min_highest_octave = f_max * (2 ** (-(bins_per_octave - 1) / bins_per_octave))
            valid_filter_length = compute_filter_length(
                f_min_highest_octave,
                sample_rate,
                bins_per_octave=bins_per_octave,
                scaling_factor=scaling_factor,
            )

            downsample = kwargs.get("downsample", aT.Resample(2, 1))

            max_filter_length = kernel.size(-1)
            trimming = max_filter_length - valid_filter_length
            trimming_left = trimming // 2
            trimming_right = trimming - trimming_left
            kernel = F.pad(kernel, (-trimming_left, -trimming_right))

            if domain == "time":
                start_bin, end_bin = 0, valid_filter_length
                n_fft = valid_filter_length
            elif domain == "freq":
                n_fft = 2 ** math.ceil(math.log2(valid_filter_length))
                # pad more samples at left for compatibility with domain == "time"
                padding = n_fft - valid_filter_length
                padding_right = padding // 2
                padding_left = padding - padding_right
                kernel = F.pad(kernel, (padding_left, padding_right))
                kernel = torch.fft.fft(kernel)

                if sparse:
                    if threshold is None:
                        threshold = 1e-3

                    kernel, start_bin, end_bin = self._trim_kernel_by_amplitude(
                        kernel, threshold=threshold
                    )
                else:
                    threshold = None
                    start_bin, end_bin = 0, n_fft
            else:
                raise ValueError(_invalid_domain_message.format(domain=domain))
        else:
            if hop_length is None:
                min_filter_length = compute_filter_length(
                    f_max,
                    sample_rate,
                    bins_per_octave=bins_per_octave,
                    scaling_factor=scaling_factor,
                )
                hop_length = min_filter_length // 4

            downsample = None
            domain = None

            # for compatibility with by_octave=True
            start_bin, end_bin = 0, kernel.size(0)
            n_fft = None

        self.register_buffer("kernel", kernel)
        self.downsample = downsample

        self.hop_length = hop_length
        self.pad = pad
        self.n_bins = n_bins
        self.center = center
        self.pad_mode = pad_mode
        self.by_octave = by_octave
        self.domain = domain
        self.sparse = sparse
        self.start_bin, self.end_bin = start_bin, end_bin
        self.n_fft = n_fft

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of CQT.

        Args:
            waveform (torch.Tensor): Input waveform of shape (\*, length).

        Returns:
            torch.Tensor: Spectrogram of CQT of shape (\*, n_bins, num_frames),
                where num_frames is determined by input length and ``hop_length``.

        """
        n_bins = self.n_bins
        by_octave = self.by_octave
        *batch_shape, length = waveform.size()

        waveform = waveform.view(-1, 1, length)

        if by_octave:
            spectrogram = self._forward_2010(waveform)
        else:
            spectrogram = self._forward_1992(waveform)

        spectrogram = spectrogram.view(*batch_shape, n_bins, -1)

        return spectrogram

    def _forward_1992(self, waveform: torch.Tensor) -> torch.Tensor:
        """Private method for forward pass of CQT1992.

        Args:
            waveform (torch.Tensor): Input waveform of shape (batch_size * in_channels, 1, length).

        Returns:
            torch.Tensor: Spectrogram of CQT of
                shape (batch_size * in_channels, n_bins, num_frames), where num_frames is
                determined by input length and ``hop_length``.

        """
        pad = self.pad
        center = self.center
        pad_mode = self.pad_mode
        max_kernel_length = self.kernel.size(-1)

        if center:
            padding = max_kernel_length // 2
            waveform = F.pad(waveform, (padding, padding), mode=pad_mode, value=pad)

        kernel = torch.view_as_real(self.kernel.unsqueeze(dim=1))
        kernel_real, kernel_imag = torch.unbind(kernel, dim=-1)

        # Since conjugate of kernel is used, -kernel_imag is convolved.
        spectrogram_real = F.conv1d(waveform, kernel_real, stride=self.hop_length)
        spectrogram_imag = F.conv1d(waveform, -kernel_imag, stride=self.hop_length)
        spectrogram = torch.stack([spectrogram_real, spectrogram_imag], dim=-1)
        spectrogram = torch.view_as_complex(spectrogram)

        return spectrogram

    def _forward_2010(self, waveform: torch.Tensor) -> torch.Tensor:
        """Private method for forward pass of CQT2010.

        Args:
            waveform (torch.Tensor): Input waveform of shape (batch_size * in_channels, 1, length).

        Returns:
            torch.Tensor: Spectrogram of CQT of shape
                (batch_size * in_channels, n_bins, num_frames), where num_frames is
                determined by input length and ``hop_length``.

        """
        n_bins = self.n_bins
        hop_length = self.hop_length
        pad = self.pad
        center = self.center
        pad_mode = self.pad_mode
        domain = self.domain
        sparse = self.sparse
        bins_per_octave = self.kernel.size(0)

        num_repeats = math.ceil(n_bins / bins_per_octave)

        spectrogram = []

        for repeat_idx in range(num_repeats):
            if repeat_idx != 0:
                waveform = self.downsample(waveform)
                hop_length //= 2

            if domain == "time":
                _spectrogram = self._temporal_forward_2010_one_octave(
                    waveform,
                    self.kernel,
                    hop_length=hop_length,
                    pad=pad,
                    center=center,
                    pad_mode=pad_mode,
                )
            elif domain == "freq":
                _spectrogram = self._spectral_forward_2010_one_octave(
                    waveform,
                    self.kernel,
                    hop_length=hop_length,
                    pad=pad,
                    center=center,
                    pad_mode=pad_mode,
                    sparse=sparse,
                    n_fft=self.n_fft,
                    start_bin=self.start_bin,
                    end_bin=self.end_bin,
                )
            else:
                raise ValueError(_invalid_domain_message.format(domain=domain))

            if repeat_idx == num_repeats - 1:
                n_remaining_bins = n_bins % bins_per_octave

                if n_remaining_bins > 0:
                    split_sections = [bins_per_octave - n_remaining_bins, n_remaining_bins]
                    _, _spectrogram = torch.split(_spectrogram, split_sections, dim=-2)

            # Since we process the lower frequencies in the next loop,
            # flip _spectrogram so that the last bin corresponds to
            # the lowest frequency at this band.
            _spectrogram = torch.flip(_spectrogram, dims=(-2,))
            spectrogram.append(_spectrogram)

        # reverse the order of frequencies
        spectrogram = torch.concat(spectrogram, dim=-2)
        spectrogram = torch.flip(spectrogram, dims=(-2,))

        return spectrogram

    @staticmethod
    def _temporal_forward_2010_one_octave(
        waveform: torch.Tensor,
        kernel: torch.Tensor,
        hop_length: int,
        pad: float = 0,
        center: bool = True,
        pad_mode: str = "reflect",
    ) -> torch.Tensor:
        """Forward pass of CQT2010 by one octave in time domain.

        Args:
            waveform (torch.Tensor): (batch_size, 1, length)
            kernel (torch.Tensor): Complex kernel of CQT of shape (bins_per_octave, kernel_length),
                where kernel_length is filter length at lowest frequency.
            hop_length (int): Hop size of CQT.
            pad (float): Value to pad. Default: ``0``.
            center (bool): Whether to pad waveform on both sides. Default: ``True``.
            pad_mode (str): Padding mode used for ``center`` is ``True``. Default: ``reflect``.

        Returns:
            torch.Tensor: CQT spectrogram of shape (batch_size, bins_per_octave, num_frames),
                where where num_frames is determined by input length and ``hop_length``.

        """
        if center:
            padding = kernel.size(-1) // 2
            waveform = F.pad(waveform, (padding, padding), mode=pad_mode, value=pad)

        kernel = torch.view_as_real(kernel.unsqueeze(dim=1))
        kernel_real, kernel_imag = torch.unbind(kernel, dim=-1)

        # Since conjugate of kernel is used, -kernel_imag is convolved.
        spectrogram_real = F.conv1d(waveform, kernel_real, stride=hop_length)
        spectrogram_imag = F.conv1d(waveform, -kernel_imag, stride=hop_length)
        spectrogram = torch.stack([spectrogram_real, spectrogram_imag], dim=-1)
        spectrogram = torch.view_as_complex(spectrogram)

        return spectrogram

    @staticmethod
    def _spectral_forward_2010_one_octave(
        waveform: torch.Tensor,
        kernel: torch.Tensor,
        hop_length: int,
        pad: float = 0,
        center: bool = True,
        pad_mode: str = "reflect",
        sparse: bool = True,
        n_fft: int = None,
        start_bin: int = 0,
        end_bin: int = -1,
    ) -> torch.Tensor:
        """Forward pass of CQT2010 by one octave in frequency domain.

        Args:
            waveform (torch.Tensor): (batch_size, 1, length)
            kernel (torch.Tensor): Complex kernel of CQT of shape (bins_per_octave, kernel_length),
                where kernel_length is filter length at lowest frequency.
            hop_length (int): Hop size of CQT.
            pad (float): Value to pad. Default: ``0``.
            center (bool): Whether to pad waveform on both sides. Default: ``True``.
            pad_mode (str): Padding mode used for ``center`` is ``True``. Default: ``reflect``.
            sparse (bool): If ``True``, kernel is treated as sparse one. Default: ``True``.

        Returns:
            torch.Tensor: CQT spectrogram of shape (batch_size, bins_per_octave, num_frames),
                where where num_frames is determined by input length and ``hop_length``.

        """
        if n_fft is None:
            n_fft = kernel.size(-1)

        if center:
            padding = n_fft // 2
            waveform = F.pad(waveform, (padding, padding), mode=pad_mode, value=pad)

        batch_size, in_channels, _ = waveform.size()

        waveform = waveform.view(batch_size * in_channels, -1)
        rectanguler_window = torch.ones(n_fft, device=waveform.device)
        spectrogram = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=rectanguler_window,
            center=False,
            pad_mode=pad_mode,
            normalized=False,
            onesided=False,
            return_complex=True,
        )

        # for negative end index
        if end_bin < 0:
            end_bin = n_fft + end_bin

        if sparse:
            _, spectrogram, _ = torch.split(
                spectrogram,
                [start_bin, end_bin - start_bin, spectrogram.size(-2) - end_bin],
                dim=-2,
            )
        else:
            assert start_bin == 0, f"'start_bin' is expected to 0, but given {start_bin}."
            assert end_bin == spectrogram.size(
                -2
            ), f"'end_bin' is expected to {spectrogram.size(-2)}, but given {end_bin}."

        # Parseval's theorem
        spectrogram = torch.matmul(kernel.conj(), spectrogram) / n_fft
        spectrogram = spectrogram.view(batch_size, in_channels, *spectrogram.size()[-2:])

        return spectrogram

    @staticmethod
    def _trim_kernel_by_amplitude(
        kernel: torch.Tensor, threshold: float
    ) -> Tuple[torch.Tensor, int, int]:
        """Trim CQT kernel by amplitude for computational efficacy.

        Args:
            kernel (torch.Tensor): Complex CQT kernel of shape (n_bins, n_fft).
            threshold (float): Threshold for trimming.

        Returns:
            tuple: Tuple containing

                - torch.Tensor: Trimmed CQT kernel of shape (n_bins, end_bin - start_bin).
                - int: Start bin index of trimmed kernel.
                - int: End bin index of trimmed kernel.

        """
        assert kernel.dim() == 2, "Dimension of kenel is expected 2, but given {}.".format(
            kernel.dim()
        )
        log_amplitude = torch.log10(torch.abs(kernel))
        non_padding_mask = log_amplitude >= log_amplitude.max() + math.log10(threshold)
        (nonzero_indices,) = torch.nonzero(non_padding_mask.sum(dim=0), as_tuple=True)
        start_bin, end_bin = torch.min(nonzero_indices), torch.max(nonzero_indices)
        start_bin, end_bin = start_bin.item(), end_bin.item() + 1
        split_sections = [start_bin, end_bin - start_bin, kernel.size(-1) - end_bin]
        _, kernel, _ = torch.split(kernel, split_sections, dim=-1)

        return kernel, start_bin, end_bin


class CQT(ConstantQTransform):
    """Alias of ConstantQTransform."""

    pass


def build_temporal_kernel(
    sample_rate: int,
    f_min: float = note_to_hz("C1"),
    f_max: Optional[float] = None,
    n_bins: int = 84,
    bins_per_octave: Optional[float] = None,
    window_fn: Optional[Callable[[int], torch.Tensor]] = torch.hann_window,
    wkwargs: Optional[dict] = None,
    scaling_factor: float = 1,
) -> torch.Tensor:
    """Build CQT temporal kernel.

    Args:
        sample_rate (int): Sampling rate.
        f_min (float): Minimum frequency. Default: ``audyn.utils.music.note_to_hz("C1")``.
        f_max (float, optional): Maximum frequency.
        n_bins (int): Number of bins. Default: ``84``.
        bins_per_octave (float, optional): Number of bins per octave.
        window_fn (callable, optional): Default: ``torch.hann_window``.
        wkwargs (dict, optional): Keyword arguments given to ``window_fn``.
        scaling_factor (float): Scaling factor. Default: ``1``.

    Returns:
        torch.Tensor: CQT temporal kernel of shape (n_bins, max_length),
            where max_length is a filter length of ``f_min``.

    .. note::

        If ``f_max`` and ``bins_per_octave`` are ``None``, ``bins_per_octave`` defaults to ``12``.
        You cannot specify ``f_max`` and ``bins_per_octave`` at same time.

    """
    f_max, bins_per_octave = _set_f_max_and_bins_per_octave(
        f_min,
        f_max=f_max,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )

    if f_max > sample_rate / 2:
        raise ValueError(_invalid_f_max_message.format(f_max=f_max, f_nyq=sample_rate / 2))

    freqs = torch.arange(n_bins)
    freqs = f_min * (2 ** (freqs / bins_per_octave))
    normalized_freqs = 2 * freqs / sample_rate

    lengths = compute_filter_length(
        freqs,
        sample_rate,
        bins_per_octave=bins_per_octave,
        scaling_factor=scaling_factor,
    )
    max_length = lengths[0].item()

    kernel = []

    for normalized_freq, length in zip(normalized_freqs, lengths):
        length = length.item()
        indices = torch.arange(length)
        _kernel = torch.exp(-1j * math.pi * normalized_freq * indices) / length

        if window_fn is not None:
            if wkwargs is None:
                kwargs = {}
            else:
                kwargs = wkwargs

            window = window_fn(length, **kwargs)
            _kernel = _kernel * window

        padding = max_length - _kernel.size(-1)
        padding_left = padding // 2
        padding_right = padding - padding_left
        _kernel = F.pad(_kernel, (padding_left, padding_right))
        kernel.append(_kernel)

    kernel = torch.stack(kernel)

    return kernel


def compute_filter_length(
    freqs: Union[float, List[float], torch.Tensor],
    sample_rate: int,
    bins_per_octave: float = 12,
    scaling_factor: float = 1,
) -> Union[int, List[int], torch.LongTensor]:
    """Compute CQT filter lengths.

    Args:
        freqs (float, list, or torch.Tensor): Sequence of analysis frequencies.
        sample_rate (int): Sampling rate.
        bins_per_octave (float, optional): Number of bins per octave. Default: ``12``.
        scaling_factor (float): Scaling factor. Default: ``1``.

    Returns:
        any: Computed filter lengths. The type is depends on that of ``freqs``.

            - If ``type(freqs)`` is ``int``, return type is ``int``.
            - If ``type(freqs)`` is ``list``, return type is ``list``.
            - If ``type(freqs)`` is ``torch.Tensor``, return type is ``torch.LongTensor``.

    """
    if isinstance(freqs, int) or isinstance(freqs, float):
        _type = "scalar"
    elif isinstance(freqs, list):
        _type = "list"
    else:
        _type = "tensor"

    freqs = torch.as_tensor(freqs)
    normalized_freqs = 2 * freqs / sample_rate
    length = (2 * scaling_factor) / (normalized_freqs * (2 ** (1 / bins_per_octave) - 1))
    length = torch.ceil(length).long()

    if _type == "scalar":
        length = length.item()
    elif _type == "list":
        length = length.tolist()

    return length


def _set_f_max_and_bins_per_octave(
    f_min: float,
    f_max: Optional[float] = None,
    n_bins: int = None,
    bins_per_octave: Optional[float] = None,
) -> Tuple[float, float]:
    if n_bins is None:
        raise ValueError("Set n_bins.")

    if bins_per_octave is None:
        if f_max is None:
            bins_per_octave = 12
            f_max = f_min * (2 ** ((n_bins - 1) / bins_per_octave))
        else:
            bins_per_octave = ((n_bins - 1) * math.log(2)) / (math.log(f_max) - math.log(f_min))
    elif f_max is None:
        f_max = f_min * (2 ** ((n_bins - 1) / bins_per_octave))
    else:
        raise ValueError("Set either f_max or bins_per_octave.")

    return f_max, bins_per_octave
