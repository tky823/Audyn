import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


def get_summary_writer(log_dir: str, is_distributed: bool = False) -> Optional[SummaryWriter]:
    """Get SummaryWriter.

    Args:
        log_dir (str): Logging directory name.
        is_distributed (bool): If ``True`` and ``os.environ["RANK"]`` is greater than ``0``,
            ``None`` is returned.

    Returns:
        SummaryWriter: Summary writer to record values.

    """
    if is_distributed and int(os.environ["RANK"]) > 0:
        writer = None
    else:
        writer = SummaryWriter(log_dir=log_dir)

    return writer
