"""PET BIDS json parsing.

Functions that take as input only the json (and not the PET data itself) are
defined here.

It might be useful to make this into its own class in the future.
"""

from __future__ import annotations

import datetime
import warnings
from copy import deepcopy
from json import dump as json_dump
from json import load as json_load
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import numpy as np

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import NDArray

    from dynamicpet.temporalobject.temporalobject import TemporalObject

# radionuclide halflives in seconds
# turkupetcentre.net/petanalysis/decay.html
HALFLIVES: dict[str, float] = {
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

    TracerRadionuclide: str

    TimeZero: str  # HH:MM:SS
    ScanStart: float  # at least one of ScanStart and InjectionStart should be 0
    InjectionStart: float
    FrameTimesStart: list[float]
    FrameDuration: list[float]
    ImageDecayCorrected: bool
    ImageDecayCorrectionTime: NotRequired[float]  # required if ImageDecayCorrected

    # entries below are not needed for any function in this module, but some are
    # required by the PET-BIDS standard
    # these tags are included below to make typeguard happy when running tests
    # using files downloaded from OpenNeuro

    # Scanner Hardware tags
    Manufacturer: NotRequired[str]
    ManufacturersModelName: NotRequired[str]
    Units: NotRequired[str]
    InstitutionName: NotRequired[str]
    InstitutionAddress: NotRequired[str]
    InstitutionalDepartmentName: NotRequired[str]
    BodyPart: NotRequired[str]

    # Radiochemistry tags
    TracerName: NotRequired[str]
    InjectedRadioactivity: NotRequired[float]
    InjectedRadioactivityUnits: NotRequired[str]
    InjectedMass: NotRequired[float | None]
    InjectedMassUnits: NotRequired[str | None]
    SpecificRadioactivity: NotRequired[float | None]
    SpecificRadioactivityUnits: NotRequired[str | None]
    ModeOfAdministration: NotRequired[str]
    TracerRadLex: NotRequired[str]
    TracerSNOMED: NotRequired[str]
    TracerMolecularWeight: NotRequired[float]
    TracerMolecularWeightUnits: NotRequired[str]
    InjectedMassPerWeight: NotRequired[float]
    InjectedMassPerWeightUnits: NotRequired[str]
    SpecificRadioactivityMeasTime: NotRequired[str]
    MolarActivity: NotRequired[float]
    MolarActivityUnits: NotRequired[str]
    MolarActivityMeasTime: NotRequired[str]
    InfusionRadioactivity: NotRequired[float]
    InfusionStart: NotRequired[float]
    InfusionSpeed: NotRequired[float]
    InfusionSpeedUnits: NotRequired[str]
    InjectedVolume: NotRequired[float]
    Purity: NotRequired[float]

    # Pharmaceuticals tags
    PharmaceuticalName: NotRequired[str]
    PharmaceuticalDoseAmount: NotRequired[float | list[float]]
    PharmaceuticalDoseUnits: NotRequired[str]
    PharmaceuticalDoseRegimen: NotRequired[str]
    PharmaceuticalDoseTime: NotRequired[float | list[float]]
    Anaesthesia: NotRequired[str]

    # Time tags
    InjectionEnd: NotRequired[float]
    ScanDate: NotRequired[str]  # DEPRECATED

    # Reconstruction tags
    AcquisitionMode: NotRequired[str]
    ReconMethodName: NotRequired[str]
    ReconMethodParameterLabels: NotRequired[list[str]]
    ReconMethodParameterUnits: NotRequired[list[str]]
    ReconMethodParameterValues: NotRequired[list[float]]
    ReconFilterType: NotRequired[str | list[str]]
    ReconFilterSize: NotRequired[float | list[float]]
    AttenuationCorrection: NotRequired[str]
    ReconMethodImplementationVersion: NotRequired[str]
    AttenuationCorrectionMethodReference: NotRequired[str]
    ScaleFactor: NotRequired[list[float]]
    ScatterFraction: NotRequired[list[float]]
    DecayCorrectionFactor: NotRequired[list[float]]
    DoseCalibrationFactor: NotRequired[float]
    PromptRate: NotRequired[list[float]]
    SinglesRate: NotRequired[list[float]]
    RandomRate: NotRequired[list[float]]

    # Task tags
    CogPOID: NotRequired[str]
    CogAtlasID: NotRequired[str]
    TaskDescription: NotRequired[str]
    Instructions: NotRequired[str]
    TaskName: NotRequired[str]


def get_hhmmss(
    json_dict: PetBidsJson,
    event: Literal[
        "ScanStart",
        "InjectionStart",
        "ImageDecayCorrectionTime",
        "FirstFrameStart",
        "TimeZero",
    ],
) -> datetime.time:
    """Get event time in HH:MM:SS.

    Args:
        json_dict: json dictionary
        event: event whose time to get in HH:MM:SS

    Returns:
        event time

    Raises:
        ValueError: image is not decay corrected

    """
    # convert TimeZero to date
    timezero = datetime.datetime.strptime(json_dict["TimeZero"], "%H:%M:%S")  # noqa: DTZ007
    if event == "TimeZero":
        return timezero.time()

    if event == "ImageDecayCorrectionTime" and not json_dict["ImageDecayCorrected"]:
        msg = "Image is not decay corrected"
        raise ValueError(msg)

    # ScanStart, InjectionStart, ImageDecayCorrectionTime, FrameTimesStart are
    # all relative to TimeZero, in seconds
    if event == "FirstFrameStart":
        offset = json_dict["FrameTimesStart"][0]
    else:
        offset = json_dict[event]
    scanstart: datetime.datetime = timezero + datetime.timedelta(seconds=offset)

    return scanstart.time()


def timediff(firsttime: datetime.time, secondtime: datetime.time) -> float:
    """Get difference in seconds between two datetime.time objects HH:MM:SS."""
    return (
        3600 * (firsttime.hour - secondtime.hour)
        + 60 * (firsttime.minute - secondtime.minute)
        + firsttime.second
        - secondtime.second
    )


def update_frametiming_from(
    json_dict: PetBidsJson,
    temporal_object: TemporalObject[Any],
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


def get_frametiming_in_mins(
    json_dict: PetBidsJson,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Get frame timing information, in minutes, from PET-BIDS json.

    PET-BIDS json must be in the 2020 format with the following tags:
    FrameDuration: Time duration of each frame in seconds
    FrameTimesStart: Start times for each frame relative to TimeZero in seconds
    ScanStart: Time of start of scan with respect to TimeZero in seconds
    InjectionStart: Time of start of injection with respect to TimeZero in
    seconds. This corresponds to DICOM Tag (0018,1042) converted to seconds
    relative to TimeZero.
    At least one of ScanStart and InjectionStart should be 0.
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
        json_dict["FrameTimesStart"],
        dtype=np.double,
    )
    frame_duration: NDArray[np.double] = np.array(
        json_dict["FrameDuration"],
        dtype=np.double,
    )

    inj_start: float = json_dict["InjectionStart"]
    scan_start: float = json_dict["ScanStart"]
    if not (inj_start == 0 or scan_start == 0):
        # invalid PET BIDS json
        msg = "Neither InjectionStart nor ScanStart is 0"
        raise ValueError(msg)

    if frame_start[0] < scan_start:
        msg = "First time frame starts before ScanStart"
        raise ValueError(msg)
    if frame_start[-1] + frame_duration[-1] < inj_start:
        warnings.warn("No data acquired after injection", stacklevel=2)

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
    if not Path(jsonfilename).exists():
        msg = f"No such file: {jsonfilename}"
        raise FileNotFoundError(msg)

    with Path(jsonfilename).open() as f:
        json_dict: PetBidsJson = json_load(f)
        return json_dict


def write_json(json_dict: PetBidsJson, filename: str | PathLike[str]) -> None:
    """Write dictionary to a json file.

    Args:
        json_dict: PET BIDS json dictionary
        filename: file name for the output

    """
    with Path(filename).open("w") as f:
        json_dump(json_dict, f, indent=4)
