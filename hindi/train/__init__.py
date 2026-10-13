# hindi/train/__init__.py
# Makes hindi/train/ a Python package so that:
#   from hindi.train.dataset import StreamingPackedDataset
# and
#   python -m hindi.train.validate_dataset
# both work when the repository root is on sys.path.

from .dataset import StreamingPackedDataset, create_dataloader
from .config import TrainingConfig
from .optimizer import create_optimizer
from .scheduler import WarmupCosineScheduler
from .evaluate import evaluate
from .checkpointing import save_checkpoint, load_checkpoint

__all__ = [
    "StreamingPackedDataset",
    "create_dataloader",
    "TrainingConfig",
    "create_optimizer",
    "WarmupCosineScheduler",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
]
