"""Patlak plotting."""

#References:
"""
https://osipi.github.io/DCE-DSC-MRI_TestResults/PatlakModel.html

https://journals.sagepub.com/doi/epdf/10.1038/jcbfm.1985.87

"""

import numpy as np

from ..temporalobject.temporalimage import TemporalImage
from ..temporalobject.temporalmatrix import TemporalMatrix
from ..typing_utils import NumpyNumberArray
from ..typing_utils import NumpyRealNumber
from .kineticmodel import KineticModel

class PATLAK(KineticModel):
    """
    
    
    
    """

    @classmethod
    def get_param_names(cls) -> list[str]:
    
        return
    
    def fit(self, mask: NumpyNumberArray[np.Any, np.dtype[np.number[np.Any]]] | None = None) -> None:
        
        """
        """
        tacs: TemporalMatrix = self.tacs.timeseries_in_mask(mask)
        reftacs: TemporalMatrix = self.reftac.timeseries_in_mask(mask)
        num_elements = tacs.num_elements
        




