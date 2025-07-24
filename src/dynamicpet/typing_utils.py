"""Type annotation utilities."""

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

NumpyRealNumber = Union[np.integer[Any], np.floating[Any]]
RealNumber = Union[int, float, NumpyRealNumber]
NumpyRealNumberArray = Union[NDArray[np.integer[Any]], NDArray[np.floating[Any]]]
NumpyNumberArray = NDArray[np.number[Any]]
