"""
Model implementations for AlphaTrader.

All models implement the BaseModel interface for plug-and-play swapping.
"""

from .base import BaseModel, ModelRegistry
from .lstm_classifier import LSTMClassifier

# xLSTM requires the xlstm package
try:
    from .xlstm_classifier import xLSTMClassifier
    __all__ = ["BaseModel", "ModelRegistry", "LSTMClassifier", "xLSTMClassifier"]
except ImportError:
    __all__ = ["BaseModel", "ModelRegistry", "LSTMClassifier"]

# Future imports:
# from .transformer import TransformerClassifier
