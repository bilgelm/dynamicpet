"""PETBIDSObject abstract class."""

from abc import ABC

import numpy as np

from ..temporalobject.temporalobject import TemporalObject
from ..typing_utils import NumpyNumberArray
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

    def get_decay_correction_factor(
        self, decaycorrecttime: float = 0
    ) -> NumpyNumberArray:
        """Get radionuclide decay correction factor.

        Args:
            decaycorrecttime: time offset (in seconds) relative to TimeZero for
                              decay correction factor calculation

        Returns:
            decay correction factors
        """
        halflife = get_radionuclide_halflife(self.json_dict)
        lmbda = np.log(2) / halflife
        lmbda_dt = lmbda * self.frame_duration
        factor = (
            -np.exp(lmbda * (self.frame_start - decaycorrecttime))
            * lmbda_dt
            / np.expm1(-lmbda_dt)
        )
        return np.array(factor)

    def get_decay_corrected_tacs(self, decaycorrecttime: float = 0) -> NumpyNumberArray:
        """Decay correct time activity curves (TACs).

        Args:
            decaycorrecttime: new time offset (in seconds) to decay correct to,
                              relative to TimeZero

        Returns:
            decay corrected TACs
        """
        # check if tacs are already decay corrected
        if (
            self.json_dict["ImageDecayCorrected"]
            and decaycorrecttime == self.json_dict["ImageDecayCorrectionTime"]
        ):
            return self.dataobj

        factor = self.get_decay_correction_factor(decaycorrecttime)
        return self.get_decay_uncorrected_tacs() * factor

    def get_decay_uncorrected_tacs(self) -> NumpyNumberArray:
        """Decay uncorrect time activity curves (TACs)."""
        # check if tacs are already decay uncorrected
        if not self.json_dict["ImageDecayCorrected"]:
            return self.dataobj

        factor = self.get_decay_correction_factor(
            self.json_dict["ImageDecayCorrectionTime"]
        )
        return self.dataobj / factor
