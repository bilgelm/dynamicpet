"""PETBIDSObject abstract class."""

from abc import ABC

import numpy as np

from ..temporalobject.temporalobject import TemporalObject
from ..typing_utils import NumpyNumberArray
from .petbidsjson import PetBidsJson
from .petbidsjson import get_hhmmss
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

    def _decay_correct_offset(self, other: "PETBIDSObject") -> float:
        """Calculate the ImageDecayCorrectionTime offset needed to match.

        Args:
            other: PETBIDSObject to be adjusted, if needed

        Returns:
            offset
        """
        if (
            self.json_dict["TracerRadionuclide"]
            != other.json_dict["TracerRadionuclide"]
        ):
            raise ValueError("Cannot concatenate data from different radionuclides")

        # check scan start times
        if get_hhmmss(self.json_dict, "ScanStart") >= get_hhmmss(
            other.json_dict, "ScanStart"
        ):
            raise ValueError("Scan times are incompatible")

        # check decay correction
        # this_decaycorrtime = get_decaycorr_rel_to_scanstart(self.json_dict)
        # other_decaycorrtime = get_decaycorr_rel_to_scanstart(other.json_dict)
        # if this_decaycorrtime != other_decaycorrtime + self.total_duration:
        #     # need to change other's decay correction to match this one's
        #     other.decay_correct(
        #       decaycorrecttime=-(other_decaycorrtime + self.total_duration)
        #     )

        this_decaycorrtime = get_hhmmss(self.json_dict, "ImageDecayCorrectionTime")
        other_decaycorrtime = get_hhmmss(other.json_dict, "ImageDecayCorrectionTime")
        offset = (
            3600 * (this_decaycorrtime.hour - other_decaycorrtime.hour)
            + 60 * (this_decaycorrtime.minute - other_decaycorrtime.minute)
            + this_decaycorrtime.second
            - other_decaycorrtime.second
        )

        return offset
