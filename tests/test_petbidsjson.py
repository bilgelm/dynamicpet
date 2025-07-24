"""Test cases for the petbidsjson file."""

# ruff: noqa: S101

import numpy as np
import pytest

from dynamicpet.petbids.petbidsjson import (
    PetBidsJson,
    get_frametiming_in_mins,
    get_hhmmss,
    get_radionuclide_halflife,
    timediff,
)


def test_get_hhmmss_scanstart0() -> None:
    """Test ScanStart, InjectionStart, ImageDecayCorrectionTime in HH:MM:SS."""
    my_json_dict: PetBidsJson = {
        "TimeZero": "10:00:00",
        "InjectionStart": -120,
        "ScanStart": 0,
        "FrameTimesStart": [60, 180, 300],
        "FrameDuration": [120, 120, 120],
        "TracerRadionuclide": "C11",
        "ImageDecayCorrected": True,
        "ImageDecayCorrectionTime": -60,
    }
    assert get_hhmmss(my_json_dict, "ScanStart").isoformat() == "10:00:00", (
        "Expected 10:00:00 ScanStart, "
        f"got {get_hhmmss(my_json_dict, 'ScanStart').isoformat()} "
    )
    assert get_hhmmss(my_json_dict, "FirstFrameStart").isoformat() == "10:01:00", (
        "Expected 10:01:00 first frame start, "
        f"got {get_hhmmss(my_json_dict, 'FirstFrameStart').isoformat()} "
    )
    assert get_hhmmss(my_json_dict, "InjectionStart").isoformat() == "09:58:00", (
        "Expected 09:58:00 InjectionStart, "
        f"got {get_hhmmss(my_json_dict, 'InjectionStart').isoformat()} "
    )
    assert (
        get_hhmmss(my_json_dict, "ImageDecayCorrectionTime").isoformat() == "09:59:00"
    ), (
        "Expected 09:59:00 image decay correction time, "
        f"got {get_hhmmss(my_json_dict, 'ImageDecayCorrectionTime').isoformat()} "
    )


def test_get_hhmmss_injstart0() -> None:
    """Test ScanStart, InjectionStart, ImageDecayCorrectionTime in HH:MM:SS."""
    my_json_dict: PetBidsJson = {
        "TimeZero": "09:58:00",
        "InjectionStart": 0,
        "ScanStart": 120,
        "FrameTimesStart": [180, 300, 420],
        "FrameDuration": [120, 120, 120],
        "TracerRadionuclide": "C11",
        "ImageDecayCorrected": True,
        "ImageDecayCorrectionTime": 60,
    }
    assert get_hhmmss(my_json_dict, "ScanStart").isoformat() == "10:00:00", (
        "Expected 10:00:00 ScanStart, "
        f"got {get_hhmmss(my_json_dict, 'ScanStart').isoformat()} "
    )
    assert get_hhmmss(my_json_dict, "FirstFrameStart").isoformat() == "10:01:00", (
        "Expected 10:01:00 first frame start, "
        f"got {get_hhmmss(my_json_dict, 'FirstFrameStart').isoformat()} "
    )
    assert get_hhmmss(my_json_dict, "InjectionStart").isoformat() == "09:58:00", (
        "Expected 09:58:00 InjectionStart, "
        f"got {get_hhmmss(my_json_dict, 'InjectionStart').isoformat()} "
    )
    assert (
        get_hhmmss(my_json_dict, "ImageDecayCorrectionTime").isoformat() == "09:59:00"
    ), (
        "Expected 09:59:00 image decay correction time, "
        f"got {get_hhmmss(my_json_dict, 'ImageDecayCorrectionTime').isoformat()} "
    )


def test_timediff() -> None:
    """Test timediff."""
    my_json_dict: PetBidsJson = {
        "TimeZero": "10:00:00",
        "InjectionStart": -120,
        "ScanStart": 0,
        "FrameTimesStart": [0, 120, 240],
        "FrameDuration": [120, 120, 120],
        "TracerRadionuclide": "C11",
        "ImageDecayCorrected": True,
        "ImageDecayCorrectionTime": -60,
    }

    msg = "Incorrect time difference"
    assert (
        timediff(
            get_hhmmss(my_json_dict, "ScanStart"),
            get_hhmmss(my_json_dict, "InjectionStart"),
        )
        == my_json_dict["ScanStart"] - my_json_dict["InjectionStart"]
    ), msg
    assert (
        timediff(
            get_hhmmss(my_json_dict, "ImageDecayCorrectionTime"),
            get_hhmmss(my_json_dict, "InjectionStart"),
        )
        == my_json_dict["ImageDecayCorrectionTime"] - my_json_dict["InjectionStart"]
    ), msg


def test_get_frametiming_from_json_scanstart0() -> None:
    """Test getting frame_start and frame_end from json when scan start is 0."""
    my_json_dict: PetBidsJson = {
        "TimeZero": "00:00:00",
        "InjectionStart": -120,
        "ScanStart": 0,
        "FrameTimesStart": [0, 120, 240],
        "FrameDuration": [120, 120, 120],
        "TracerRadionuclide": "C11",
        "ImageDecayCorrected": True,
    }
    frame_start, frame_duration = get_frametiming_in_mins(my_json_dict)
    assert np.all(frame_start == np.array([0, 120, 240]) / 60), (
        "Mismatch in frame starts"
    )
    assert np.all(frame_duration == np.array([120, 120, 120]) / 60), (
        "Mismatch in frame durations"
    )


def test_get_frametiming_from_invalid_jsons() -> None:
    """Test getting frame_start and frame_end from invalid json."""
    my_json_dict: PetBidsJson = {
        "TimeZero": "00:00:00",
        "InjectionStart": 1,
        "ScanStart": 1,
        "FrameTimesStart": [0, 120, 240],
        "FrameDuration": [120, 120, 120],
        "TracerRadionuclide": "C11",
        "ImageDecayCorrected": True,
    }

    with pytest.raises(
        ValueError,
        match="Neither InjectionStart nor ScanStart is 0",
    ):
        get_frametiming_in_mins(my_json_dict)


def test_halflife() -> None:
    """Test halflife for C11."""
    my_json_dict: PetBidsJson = {
        "TimeZero": "00:00:00",
        "InjectionStart": -120,
        "ScanStart": 0,
        "FrameTimesStart": [0, 120, 240],
        "FrameDuration": [120, 120, 120],
        "TracerRadionuclide": "C11",
        "ImageDecayCorrected": True,
    }
    assert get_radionuclide_halflife(my_json_dict) == 1224 / 60, (
        f"Expected {1224 / 60} min halflife, ",
        "got {get_radionuclide_halflife(my_json_dict)}",
    )
