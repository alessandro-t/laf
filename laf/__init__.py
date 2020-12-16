__version__ = '1.0.0'

from .model_torch import (
    LAFLayer, LAFLayerFast
)
from .model_tf import (
    LAFLayerTF, LAFLayerFastTF
)
__all__ = [
    'LAFLayer', 'LAFLayerFast', 'LAFLayerTF', 'LAFLayerFastTF'
]
