"""Calculate standardized update value ratio (SUVR)."""

from typing import Any

import numpy as np

from dynamicpet.temporalobject.temporalobject import TemporalObject

from ..typing_utils import NumpyRealNumber
from ..typing_utils import NumpyRealNumberArray
from .kineticmodel import KineticModel


class SUVR(KineticModel):
    """Standardized uptake value ratio (SUVR).

    SUVR = SUV (target) / SUV (reference)
         = frame duration weighted sum of target TAC /
            frame duration weighted sum of reference TAC

    Before initializing the SUVR object, make sure to extract the time window
    over which you wish to calculate SUVR.
    """

    def fit(self) -> None:
        """Calculate SUVR.

        Example:
            >>> import numpy as np
            >>> from dynamicpet.temporalobject import TemporalMatrix
            >>> from dynamicpet.kineticmodel.suvr import SUVR
            >>> frame_start = [60, 70]
            >>> frame_duration = [10, 10]
            >>> reftac = TemporalMatrix(dataobj=np.array([2, 2]),
                                        frame_start=frame_start,
                                        frame_duration=frame_duration)
            >>> tacs = TemporalMatrix(dataobj=np.array([[3, 3], [6, 6]]),
                                      frame_start=frame_start,
                                      frame_duration=frame_duration)
            >>> km = SUVR(reftac, tacs)
            >>> km.fit()
            >>> km.get_parameter('suvr')
            array([1.5, 3. ])
        """
        numerator: NumpyRealNumberArray = np.sum(
            self.tacs.dataobj * self.tacs.frame_duration, axis=-1
        )
        denominator: NumpyRealNumber = np.sum(
            self.reftac.dataobj * self.reftac.frame_duration
        )
        suvr = numerator / denominator

        self.parameters["suvr"] = suvr

    def fitted_tacs(self) -> TemporalObject[Any]:
        """Get fitted TACs based on estimated model parameters."""
        # there is no parametric model for SUVR, so we just return the tacs
        return self.tacs
