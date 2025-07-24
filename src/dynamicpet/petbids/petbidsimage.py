"""PETBIDSImage class."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from nibabel.loadsave import load as nib_load

from dynamicpet.petbids.petbidsjson import (
    PetBidsJson,
    get_frametiming_in_mins,
    read_json,
    update_frametiming_from,
    write_json,
)
from dynamicpet.petbids.petbidsobject import PETBIDSObject
from dynamicpet.temporalobject.temporalimage import TemporalImage, image_maker

if TYPE_CHECKING:
    from os import PathLike

    from nibabel.spatialimages import SpatialImage

    from dynamicpet.typing_utils import RealNumber


class PETBIDSImage(TemporalImage, PETBIDSObject):
    """4-D image data with corresponding PET-BIDS time frame information.

    Args:
        img: a SpatialImage object with a 3-D or 4-D dataobj
        json_dict: PET-BIDS json dictionary

    Attributes:
        img: SpatialImage storing image data matrix and header
        frame_start: vector containing the start times of each frame, in min
        frame_duration: vector containing the durations of each frame, in min
        json_dict: PET-BIDS json dictionary

    """

    def __init__(self, img: SpatialImage, json_dict: PetBidsJson) -> None:
        """4-D image data with corresponding PET-BIDS time frame information.

        Args:
            img: a SpatialImage object with a 3-D or 4-D dataobj
            json_dict: PET-BIDS json dictionary

        """
        frame_start, frame_duration = get_frametiming_in_mins(json_dict)

        super().__init__(img, frame_start, frame_duration)

        # need to make a copy of json_dict before storing
        self.json_dict: PetBidsJson = deepcopy(json_dict)

    def extract(self, start_time: RealNumber, end_time: RealNumber) -> PETBIDSImage:
        """Extract a temporally shorter PETBIDSImage from a PETBIDSImage.

        Args:
            start_time: time (min) at which to begin relative to TimeZero, incl.
            end_time: time (min) at which to stop relative to TimeZero, incl.

        Returns:
            extracted PETBIDSImage

        """
        extracted_img = super().extract(start_time, end_time)
        json_dict = update_frametiming_from(self.json_dict, extracted_img)

        return PETBIDSImage(extracted_img.img, json_dict)

    def concatenate(self, other: PETBIDSImage) -> PETBIDSImage:  # type: ignore[override]
        """Concatenate another PETBIDSImage at the end (in time).

        Args:
            other: PETBIDSImage to concatenate

        Returns:
            concatenated PETBIDSImage

        """
        newdecaycorrecttime, original_anchor = self._decay_correct_offset(other)
        other = other.decay_correct(decaycorrecttime=newdecaycorrecttime)

        concat_img = super().concatenate(other)
        json_dict = update_frametiming_from(self.json_dict, concat_img)

        concat_res = PETBIDSImage(concat_img.img, json_dict)
        concat_res.set_timezero(anchor=original_anchor)

        return concat_res

    def decay_correct(self, decaycorrecttime: float = 0) -> PETBIDSImage:
        """Return decay corrected PETBIDSImage.

        This code is written to work with both ScanStart and InjectionStart as
        TimeZero anchors, even though the internal representation is always
        with an InjectionStart anchor.

        Args:
            decaycorrecttime: time to decay correct to, relative to time zero

        Returns:
            decay corrected PET image

        """
        tacs = self.get_decay_corrected_tacs(decaycorrecttime)
        # Create a SpatialImage of the same class as self.img
        uncorrected_img = image_maker(np.reshape(tacs, self.shape), self.img)

        json_dict = deepcopy(self.json_dict)
        json_dict["ImageDecayCorrected"] = True
        json_dict["ImageDecayCorrectionTime"] = (
            decaycorrecttime + json_dict["ScanStart"] + json_dict["InjectionStart"]
        )

        return PETBIDSImage(uncorrected_img, json_dict)

    def decay_uncorrect(self) -> PETBIDSImage:
        """Return decay uncorrected PETBIDSImage."""
        tacs = self.get_decay_uncorrected_tacs()
        # Create a SpatialImage of the same class as self.img
        uncorrected_img = image_maker(np.reshape(tacs, self.shape), self.img)

        json_dict = deepcopy(self.json_dict)
        json_dict["ImageDecayCorrected"] = False
        # PET-BIDS still requires "ImageDecayCorrectionTime" tag, so we don't
        # do anything about it

        return PETBIDSImage(uncorrected_img, json_dict)

    def to_filename(
        self,
        filename: str | PathLike[str],
        anchor: Literal["InjectionStart", "ScanStart"] = "InjectionStart",
        *,
        save_json: bool = False,
    ) -> None:
        """Save to file.

        Args:
            filename: file name for the PET image output
            save_json: whether the PET-BIDS json side car should be saved
            anchor: time anchor. The corresponding tag in the PET-BIDS json will
                    be set to zero (with appropriate offsets applied to other
                    tags).

        """
        self.img.to_filename(filename)

        if save_json:
            self.set_timezero(anchor)

            fname = Path(filename)
            if fname.suffix == ".gz":
                jsonfilename = fname.with_suffix("").with_suffix(".json")
            else:
                jsonfilename = fname.with_suffix(".json")

            write_json(self.json_dict, jsonfilename)


def load(
    filename: str | PathLike[str],
    jsonfilename: str | PathLike[str] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> PETBIDSImage:
    """Load a PET image and accompanying BIDS json.

    Args:
        filename: path to 4-D image file to load
        jsonfilename: path to PET-BIDS json file with frame timing info
        kwargs: keyword arguments to format-specific load (see nibabel.load)

    Returns:
        loaded image

    Raises:
        FileNotFoundError: filename or jsonfilename was not found

    """
    fname = Path(filename)
    if not fname.exists():
        msg = f"No such file: {filename}"
        raise FileNotFoundError(msg)

    if jsonfilename is None:
        if fname.suffix == ".gz":
            jsonfilename = fname.with_suffix("").with_suffix(".json")
        else:
            jsonfilename = fname.with_suffix(".json")

    img: SpatialImage = nib_load(filename, **kwargs)  # type: ignore[assignment]

    json_dict: PetBidsJson = read_json(jsonfilename)

    return PETBIDSImage(img, json_dict)
