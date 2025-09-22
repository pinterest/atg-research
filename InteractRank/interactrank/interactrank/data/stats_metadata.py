from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

class FeatureStatsMetadata:
    """
    The FeatureStats metadata.

    :param min_value: The minimum value of the feature.
    :param max_value: The maximum value of the feature.
    :param mean: The mean value of the feature.
    :param std: The standard deviation of the feature.
    :param vocab: The vocabulary of the feature values [applicable for sparse/categorical features only].
    """

    # min_value: Optional[Union[float, List[float]]] = 0
    # max_value: Optional[Union[float, List[float]]] = 0
    # mean: Optional[Union[float, List[float]]] = 0
    # std: Optional[Union[float, List[float]]] = 0
    min_value: Optional[float] = 0
    max_value: Optional[float] = 0
    mean: Optional[float] = 0
    std: Optional[float] = 0
    vocab: Optional[Dict[int, int]] = None
    signal_group: Optional[str] = None
    signal_type: Optional[str] = None