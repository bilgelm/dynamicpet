"""PETBIDSImage class."""

import os.path as op
from copy import deepcopy
from os import PathLike
from typing import Any

import numpy as np
from nibabel.loadsave import load as nib_load
from nibabel.spatialimages import SpatialImage

from ..temporalobject.temporalimage import TemporalImage
from ..typing_utils import RealNumber
from .petbidsjson import PetBidsJson
from .petbidsjson import get_frametiming
from .petbidsjson import read_json
from .petbidsjson import update_frametiming_from
from .petbidsobject import PETBIDSObject


class PETBIDSImage(TemporalImage, PETBIDSObject):
    """4D image data with corresponding PET-BIDS time frame information.

    Attributes:
        img: SpatialImage storing image data matrix and header
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing the durations of each frame
        json_dict: PET-BIDS json dictionary
    """

    def __init__(self, img: SpatialImage, json_dict: PetBidsJson) -> None:
        """4D image data with corresponding PET-BIDS time frame information.

        Args:
            img: a SpatialImage object with a 3D or 4D dataobj
            json_dict: PET-BIDS json dictionary
        """
        frame_start, frame_duration = get_frametiming(json_dict)

        super().__init__(img, frame_start, frame_duration)

        # need to make a copy of json_dict before storing
        self.json_dict: PetBidsJson = deepcopy(json_dict)

    def extract(self, start_time: RealNumber, end_time: RealNumber) -> "PETBIDSImage":
        """Extract a PETBIDSImage from a temporally longer PETBIDSImage.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, exclusive

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
        """
        if self.json_dict["TracerNuclide"] != other.json_dict["TracerNuclide"]:
            raise ValueError("Cannot concatenate data from different radionuclides")

        concat_img = super().concatenate(other)
        json_dict = update_frametiming_from(self.json_dict, concat_img)

        concat_res = PETBIDSImage(concat_img.img, json_dict)

        return concat_res

    def decay_correct(self) -> "PETBIDSImage":
        """Return decay corrected PETBIDSImage."""
        tacs = self.get_decay_corrected_tacs()
        # Create a SpatialImage of the same class as self.img
        image_maker = self.img.__class__
        corrected_img = image_maker(
            np.reshape(tacs, self.shape), self.img.affine, self.img.header
        )

        return PETBIDSImage(corrected_img, self.json_dict)

    def decay_uncorrect(self) -> "PETBIDSImage":
        """Return decay uncorrected PETBIDSImage."""
        tacs = self.get_decay_uncorrected_tacs()
        # Create a SpatialImage of the same class as self.img
        image_maker = self.img.__class__
        corrected_img = image_maker(
            np.reshape(tacs, self.shape), self.img.affine, self.img.header
        )

        return PETBIDSImage(corrected_img, self.json_dict)


def load(
    filename: str | PathLike[str],
    jsonfilename: str | PathLike[str] | None = None,
    **kwargs: Any
) -> PETBIDSImage:
    """Load a PET image and accompanying BIDS json.

    Args:
        filename: path to 4D image file to load
        jsonfilename: path to csv file containing frame timing information
        kwargs: Keyword arguments to format-specific load (see nibabel.load)

    Returns:
        temporal image object

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
