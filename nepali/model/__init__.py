"""
nepali/model/__init__.py
=======================
Public API for the Nepali decoder-only Transformer model.

Re-exports the two most commonly needed symbols so users can write:

    from nepali.model import DecoderOnlyTransformer
    from nepali.model import ModelConfig

instead of importing from the sub-modules directly.

Kaggle usage example
--------------------
    import sys
    sys.path.insert(0, "/kaggle/working/individual-project-charan-1230")

    from nepali.model import DecoderOnlyTransformer, ModelConfig

    cfg   = ModelConfig.from_yaml("nepali/configs/model_config.yaml")
    model = DecoderOnlyTransformer(cfg)
    model.eval()

Nepali reuse
------------
Copy the entire nepali/model/ directory to nepali/model/ (unchanged).
Then import with:

    from nepali.model import DecoderOnlyTransformer, ModelConfig
    cfg = ModelConfig.from_yaml("nepali/configs/model_config.yaml")
"""

from .config import ModelConfig
from .transformer import DecoderOnlyTransformer

__all__ = [
    "ModelConfig",
    "DecoderOnlyTransformer",
]
