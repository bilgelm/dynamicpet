"""PETBIDSObject abstract class."""

from abc import ABC

import numpy as np

from ..temporalobject.temporalobject import TemporalObject
from ..typing_utils import NumpyRealNumberArray
from .petbidsjson import PetBidsJson
from .petbidsjson import get_radionuclide_halflife


class PETBIDSObject(TemporalObject["PETBIDSObject"], ABC):
    """PETBIDSObject abstract base class.

    Attributes:
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing the durations of each frame
        json_dict: PET-BIDS json dictionary
    """

    json_dict: PetBidsJson

    def get_decay_correction_factor(self) -> NumpyRealNumberArray:
        """Get radionuclide decay correction factor."""
        halflife = get_radionuclide_halflife(self.json_dict)
        lmbda = np.log(2) / halflife
        lmbda_dt = lmbda * self.frame_duration
        factor = -np.exp(lmbda * self.frame_start) * lmbda_dt / np.expm1(-lmbda_dt)
        return np.array(factor)

    def get_decay_corrected_tacs(self) -> NumpyRealNumberArray:
        """Decay correct time activity curves to time zero."""
        # check if tacs are already decay corrected
        # TODO
        factor = self.get_decay_correction_factor()
        return self.dataobj * factor

    def get_decay_uncorrected_tacs(self) -> NumpyRealNumberArray:
        """Decay uncorrect TACs (assuming decay correction was to time zero)."""
        # check if tacs are already decay uncorrected
        # TODO
        factor = self.get_decay_correction_factor()
        return self.dataobj / factor
