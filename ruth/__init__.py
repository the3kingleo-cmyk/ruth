"""Ruth: a token-free, continuous-time mind that learns while she lives.

No tokens, no vocabulary, no context window, no API keys, no plugins, no
pretrained model: a brain built from continuous dynamics, and a mind that
thinks, keeps confidences, dreams and grows a temperament of her own.
"""
from .config import BrainConfig
from .engine import Brain

__all__ = ["Brain", "BrainConfig"]
__version__ = "2.2.0"
