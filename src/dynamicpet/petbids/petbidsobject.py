"""PETBIDSObject abstract class."""

from abc import ABC
from typing import Literal

import numpy as np

from dynamicpet.petbids.petbidsjson import (
    PetBidsJson,
    get_hhmmss,
    get_radionuclide_halflife,
    timediff,
)
from dynamicpet.temporalobject.temporalobject import TemporalObject
from dynamicpet.typing_utils import NumpyNumberArray


class PETBIDSObject(TemporalObject["PETBIDSObject"], ABC):
    """PETBIDSObject abstract base class.

    Attributes:
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing the durations of each frame
        json_dict: PET-BIDS json dictionary

    """

    json_dict: PetBidsJson

    def set_timezero(
        self,
        anchor: Literal["InjectionStart", "ScanStart"] = "InjectionStart",
    ) -> None:
        """Modify time tags and frame time start to specified anchor."""
        if self.json_dict[anchor] == 0:
            return

        offset = self.json_dict[anchor]  # offset in seconds
        frametimesstart = self.json_dict["FrameTimesStart"]
        new_timezero = get_hhmmss(self.json_dict, anchor)

        # update attribute
        self.frame_start = self.frame_start - offset / 60

        # update json tags
        scalar_tags_to_update = [
            "InjectionStart",
            "ScanStart",
            "ImageDecayCorrectionTime",
            "InjectionEnd",
        ]
        for tag in scalar_tags_to_update:
            if tag in self.json_dict:
                self.json_dict[tag] -= offset  # type: ignore[literal-required]

        self.json_dict["FrameTimesStart"] = [fts - offset for fts in frametimesstart]
        self.json_dict["TimeZero"] = new_timezero.strftime("%H:%M:%S")

    def get_decay_correction_factor(
        self,
        decaycorrecttime: float = 0,
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
            -np.exp(lmbda * (self.frame_start - decaycorrecttime / 60))
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
            self.json_dict["ImageDecayCorrectionTime"],
        )
        return self.dataobj / factor

    def _decay_correct_offset(
        self,
        other: "PETBIDSObject",
    ) -> tuple[float, Literal["InjectionStart", "ScanStart"]]:
        """Calculate ImageDecayCorrectionTime offset needed to match other to self.

        This is a helper function for concatenate.

        Args:
            other: PETBIDSObject to be adjusted, if needed

        Returns:
            newdecaycorrecttime: new decay correction time relative to time zero
                                 of self, in seconds
            original_anchor: anchor time of self

        Raises:
            ValueError: radionuclides or injection/scan times are incompatible

        """
        # check if scans are combineable
        # - verify same radionuclide
        if (
            self.json_dict["TracerRadionuclide"]
            != other.json_dict["TracerRadionuclide"]
        ):
            msg = "Radionuclides are incompatible"
            raise ValueError(msg)

        # - verify same injection time
        if get_hhmmss(self.json_dict, "InjectionStart") != get_hhmmss(
            other.json_dict,
            "InjectionStart",
        ):
            msg = "Injection times are incompatible"
            raise ValueError(msg)

        # - check scan timing
        this_firstframestart = get_hhmmss(self.json_dict, "FirstFrameStart")
        other_firstframestart = get_hhmmss(other.json_dict, "FirstFrameStart")
        if timediff(other_firstframestart, this_firstframestart) <= self.total_duration:
            msg = "Scan times are incompatible"
            raise ValueError(msg)

        original_anchor: Literal["InjectionStart", "ScanStart"]
        if self.json_dict["InjectionStart"] == 0:
            original_anchor = "InjectionStart"
        else:
            original_anchor = "ScanStart"
            self.set_timezero(anchor="InjectionStart")

        other.set_timezero(anchor="InjectionStart")

        newdecaycorrecttime = self.json_dict["ImageDecayCorrectionTime"]
        return newdecaycorrecttime, original_anchor
