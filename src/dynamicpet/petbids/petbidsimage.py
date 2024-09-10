"""PETBIDSImage class."""

import os.path as op
from copy import deepcopy
from os import PathLike
from typing import Any

import numpy as np
from nibabel.loadsave import load as nib_load
from nibabel.spatialimages import SpatialImage

from ..temporalobject.temporalimage import TemporalImage
from ..temporalobject.temporalimage import image_maker
from ..typing_utils import RealNumber
from .petbidsjson import PetBidsJson
from .petbidsjson import get_frametiming_in_mins
from .petbidsjson import get_hhmmss
from .petbidsjson import read_json
from .petbidsjson import update_frametiming_from
from .petbidsjson import write_json
from .petbidsobject import PETBIDSObject


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

    def extract(self, start_time: RealNumber, end_time: RealNumber) -> "PETBIDSImage":
        """Extract a temporally shorter PETBIDSImage from a PETBIDSImage.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, inclusive

        Returns:
            extracted_img: extracted PETBIDSImage
        """
        extracted_img = super().extract(start_time, end_time)
        json_dict = update_frametiming_from(self.json_dict, extracted_img)

        extract_res = PETBIDSImage(extracted_img.img, json_dict)
        return extract_res

    def concatenate(self, other: "PETBIDSImage") -> "PETBIDSImage":  # type: ignore
        """Concatenate another PETBIDSImage at the end (in time).

        Args:
            other: PETBIDSImage to concatenate

        Returns:
            concatenated PETBIDSImage

        Raises:
            ValueError: PETBIDSImages are from different radionuclides
            NotImplementedError: decay correction times are different
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
        if this_decaycorrtime != other_decaycorrtime:
            # need to change other's decay correction to match this one's
            offset = (
                3600 * (this_decaycorrtime.hour - other_decaycorrtime.hour)
                + 60 * (this_decaycorrtime.minute - other_decaycorrtime.minute)
                + this_decaycorrtime.second
                - other_decaycorrtime.second
            )
            other.decay_correct(decaycorrecttime=offset)

        concat_img = super().concatenate(other)
        json_dict = update_frametiming_from(self.json_dict, concat_img)

        concat_res = PETBIDSImage(concat_img.img, json_dict)

        return concat_res

    def decay_correct(self, decaycorrecttime: float = 0) -> "PETBIDSImage":
        """Return decay corrected PETBIDSImage.

        Args:
            decaycorrecttime: time to decay correct to, relative to time zero
        """
        tacs = self.get_decay_corrected_tacs(decaycorrecttime)
        # Create a SpatialImage of the same class as self.img
        corrected_img = image_maker(np.reshape(tacs, self.shape), self.img)

        json_dict = self.json_dict
        json_dict["ImageDecayCorrected"] = True
        json_dict["ImageDecayCorrectionTime"] = (
            decaycorrecttime + json_dict["ScanStart"] + json_dict["InjectionStart"]
        )

        return PETBIDSImage(corrected_img, json_dict)

    def decay_uncorrect(self) -> "PETBIDSImage":
        """Return decay uncorrected PETBIDSImage."""
        tacs = self.get_decay_uncorrected_tacs()
        # Create a SpatialImage of the same class as self.img
        corrected_img = image_maker(np.reshape(tacs, self.shape), self.img)

        json_dict = self.json_dict
        json_dict["ImageDecayCorrected"] = False
        # PET-BIDS still requires "ImageDecayCorrectionTime" tag, so we don't
        # do anything about it

        return PETBIDSImage(corrected_img, json_dict)

    def to_filename(self, filename: str | PathLike[str]) -> None:
        """Save to file.

        Args:
            filename: file name for the PET image output
        """
        self.img.to_filename(filename)

        fbase, fext = op.splitext(filename)
        if fext == ".gz":
            fbase = op.splitext(fbase)[0]
        jsonfilename = fbase + ".json"

        write_json(self.json_dict, jsonfilename)


def load(
    filename: str | PathLike[str],
    jsonfilename: str | PathLike[str] | None = None,
    **kwargs: Any
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
    if not op.exists(filename):
        raise FileNotFoundError("No such file: '%s'" % filename)

    if jsonfilename is None:
        jsonbase, jsonext = op.splitext(filename)
        if jsonext == ".gz":
            jsonbase = op.splitext(jsonbase)[0]
        jsonfilename = jsonbase + ".json"

    img: SpatialImage = nib_load(filename, **kwargs)  # type: ignore
    # img = SpatialImage.from_filename(filename, **kwargs)

    json_dict: PetBidsJson = read_json(jsonfilename)

    ti = PETBIDSImage(img, json_dict)
    return ti
