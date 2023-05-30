"""PET BIDS json parsing.

Functions that take as input only the json (and not the PET data itself) are
defined here.

It might be useful to make this into its own class in the future.
"""

import os.path as op
import warnings
from copy import deepcopy
from json import load as json_load
from os import PathLike
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from ..temporalobject.temporalobject import TemporalObject


# radionuclide halflives in seconds
# turkupetcentre.net/petanalysis/decay.html
HALFLIVES: Dict[str, float] = {
    "c11": 1224,
    "n13": 599,
    "o15": 123,
    "f18": 6588,
    "cu62": 582,
    "cu64": 45721.1,
    "ga68": 4080,
    "ge68": 23760000,
    "br76": 58700,
    "rb82": 75,
    "zr89": 282240,
    "i124": 360806.4,
}


class PetBidsJson(TypedDict):
    """PET-BIDS json dictionary."""

    InjectionStart: float
    ScanStart: float
    FrameTimesStart: List[float]
    FrameDuration: List[float]

    TracerRadionuclide: str


def update_frametiming_from(
    json_dict: PetBidsJson, temporal_object: TemporalObject[Any]
) -> PetBidsJson:
    """Update frame timing information in PET-BIDS json from TemporalObject.

    Args:
        json_dict: json dictionary to be updated
        temporal_object: TemporalObject to pull frame timing info from

    Returns:
        updated json dictionary
    """
    new_json_dict: PetBidsJson = deepcopy(json_dict)
    # convert from minutes to seconds
    new_json_dict["FrameTimesStart"] = (temporal_object.frame_start * 60).tolist()
    new_json_dict["FrameDuration"] = (temporal_object.frame_duration * 60).tolist()
    return new_json_dict


def get_frametiming(
    json_dict: PetBidsJson,
) -> Tuple[NDArray[np.double], NDArray[np.double]]:
    """Get frame timing information, in minutes, from PET-BIDS json.

    PET-BIDS json must be in the 2020 format with the following tags:
    FrameDuration: Time duration of each frame in seconds
    FrameTimesStart: Start times for each frame relative to TimeZero in seconds
    ScanStart: Time of start of scan with respect to TimeZero in seconds
    InjectionStart: Time of start of injection with respect to TimeZero in
                    seconds. This corresponds to DICOM Tag (0018,1042)
                    converted to seconds relative to TimeZero.
    At least one of ScanStart and InjectionStart should be 0.
    If ScanStart is 0, FrameTimesStart are shifted so that outputs are relative
    to injection start.
    This method does not check if the FrameTimesStart and FrameDuration entries
    in the json file are sensible.

    Args:
        json_dict: PET-BIDS json dictionary

    Returns:
        frame_start: vector of frame start times relative to injection start, in minutes
        frame_end: vector of frame end times relative to injection start, in minutes

    Raises:
        ValueError: invalid frame timing
    """
    frame_start: NDArray[np.double] = np.array(
        json_dict["FrameTimesStart"], dtype=np.double
    )
    frame_duration: NDArray[np.double] = np.array(
        json_dict["FrameDuration"], dtype=np.double
    )

    inj_start: float = json_dict["InjectionStart"]
    scan_start: float = json_dict["ScanStart"]
    if inj_start == 0:
        pass
    elif scan_start == 0:
        if frame_start[-1] + frame_duration[-1] < inj_start:
            warnings.warn("No data acquired after injection", stacklevel=2)
        frame_start -= inj_start
    else:
        # invalid PET BIDS json
        raise ValueError("Neither InjectionStart nor ScanStart is 0")

    # convert seconds to minutes
    return frame_start / 60, frame_duration / 60


def get_radionuclide_halflife(json_dict: PetBidsJson) -> float:
    """Get halflife of radionuclide from PET BIDS JSON dictionary."""
    radionuclide = json_dict["TracerRadionuclide"].lower().replace("-", "")

    # convert seconds to minutes
    return HALFLIVES[radionuclide] / 60


def read_json(jsonfilename: str | PathLike[str]) -> PetBidsJson:
    """Read PET-BIDS json (no validity checks).

    Args:
        jsonfilename: path to csv file containing frame timing information

    Returns:
        json dictionary

    Raises:
        FileNotFoundError: jsonfilename was not found
    """
    if not op.exists(jsonfilename):
        raise FileNotFoundError("No such file: '%s'" % jsonfilename)

    with open(jsonfilename) as f:
        json_dict: PetBidsJson = json_load(f)
        return json_dict
