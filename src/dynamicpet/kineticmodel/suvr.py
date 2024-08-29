"""Standardized update value ratio (SUVR)."""

import numpy as np

from ..temporalobject.temporalimage import TemporalImage
from ..temporalobject.temporalmatrix import TemporalMatrix
from ..typing_utils import NumpyNumberArray
from ..typing_utils import NumpyRealNumber
from .kineticmodel import KineticModel


class SUVR(KineticModel):
    """Standardized uptake value ratio (SUVR).

    SUVR = SUV (target) / SUV (reference)
         = frame duration weighted sum of target TAC /
            frame duration weighted sum of reference TAC
    """

    @classmethod
    def get_param_names(cls) -> list[str]:
        """Get names of kinetic model parameters."""
        return ["suvr"]

    def fit(self, mask: NumpyNumberArray | None = None) -> None:
        """Calculate SUVR.

        Args:
            mask: [optional] A 1-D (for TemporalMatrix TACs) or
                  3-D (for TemporalImage TACs) binary mask that defines where
                  to fit the kinetic model. Elements outside the mask will
                  be set to to 0 in parametric estimate outputs.

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

        numerator: NumpyNumberArray = np.sum(
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
