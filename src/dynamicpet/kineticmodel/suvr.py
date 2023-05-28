"""Calculate standardized update value ratio (SUVR)."""

from typing import List

import numpy as np

from dynamicpet.temporalobject.temporalimage import TemporalImage
from dynamicpet.temporalobject.temporalmatrix import TemporalMatrix

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

    @classmethod
    def get_param_names(cls) -> List[str]:
        """Get names of kinetic model parameters."""
        return ["suvr"]

    def fit(self, mask: NumpyRealNumberArray | None = None) -> None:
        """Calculate SUVR.

        Args:
            mask: an optional parameter used only when tacs attribute is a
                  TemporalImage (or inherits from TemporalImage). A 3-D binary
                  mask that defines where to fit the kinetic model. Voxels
                  outside the mask will be set to NA in output parametric images.

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
        tacs: TemporalMatrix = self.tacs.timeseries_in_mask(mask)

        numerator: NumpyRealNumberArray = np.sum(
            tacs.dataobj * tacs.frame_duration, axis=-1
        )
        denominator: NumpyRealNumber = np.sum(
            self.reftac.dataobj * self.reftac.frame_duration
        )
        suvr = numerator / denominator

        self.set_parameter("suvr", suvr, mask)

    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        """Get fitted TACs based on estimated model parameters."""
        # there is no parametric model for SUVR, so we just return the tacs
        return self.tacs
