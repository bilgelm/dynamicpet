"""Type annotation utilities."""

from typing import Any
from typing import Union

import numpy as np
from numpy.typing import NDArray


NumpyRealNumber = Union[np.unsignedinteger[Any], np.integer[Any], np.floating[Any]]
RealNumber = Union[int, float, NumpyRealNumber]
NumpyRealNumberArray = NDArray[NumpyRealNumber]
