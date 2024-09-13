"""Test cases for the petbidsjson file."""

import numpy as np
import pytest

from dynamicpet.petbids.petbidsjson import PetBidsJson
from dynamicpet.petbids.petbidsjson import get_frametiming_in_mins
from dynamicpet.petbids.petbidsjson import get_hhmmss
from dynamicpet.petbids.petbidsjson import get_radionuclide_halflife
from dynamicpet.petbids.petbidsjson import timediff


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
    assert "10:00:00" == get_hhmmss(my_json_dict, "ScanStart").isoformat()
    assert "10:01:00" == get_hhmmss(my_json_dict, "FirstFrameStart").isoformat()
    assert "09:58:00" == get_hhmmss(my_json_dict, "InjectionStart").isoformat()
    assert (
        "09:59:00" == get_hhmmss(my_json_dict, "ImageDecayCorrectionTime").isoformat()
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
    assert "10:00:00" == get_hhmmss(my_json_dict, "ScanStart").isoformat()
    assert "10:01:00" == get_hhmmss(my_json_dict, "FirstFrameStart").isoformat()
    assert "09:58:00" == get_hhmmss(my_json_dict, "InjectionStart").isoformat()
    assert (
        "09:59:00" == get_hhmmss(my_json_dict, "ImageDecayCorrectionTime").isoformat()
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
    assert (
        timediff(
            get_hhmmss(my_json_dict, "ScanStart"),
            get_hhmmss(my_json_dict, "InjectionStart"),
        )
        == my_json_dict["ScanStart"] - my_json_dict["InjectionStart"]
    )
    assert (
        timediff(
            get_hhmmss(my_json_dict, "ImageDecayCorrectionTime"),
            get_hhmmss(my_json_dict, "InjectionStart"),
        )
        == my_json_dict["ImageDecayCorrectionTime"] - my_json_dict["InjectionStart"]
    )


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
    assert np.all(frame_start == np.array([0, 120, 240]) / 60)
    assert np.all(frame_duration == np.array([120, 120, 120]) / 60)


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

    with pytest.raises(ValueError) as excinfo:
        get_frametiming_in_mins(my_json_dict)
    assert str(excinfo.value) == "Neither InjectionStart nor ScanStart is 0"
    # my_json_dict: PetBidsJson = {
    #     "InjectionStart": 0,
    #     "ScanStart": 0,
    #     "FrameTimesStart": [999, 120, 240],
    #     "FrameDuration": [120, 120, 120],
    #     "TracerRadionuclide": "C11",
    # }
    # with pytest.raises(TimingError) as excinfo:
    #     get_frametiming_in_mins(my_json_dict)
    # assert "Non-increasing frame start times" in str(excinfo.value)


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
    assert get_radionuclide_halflife(my_json_dict) == 1224 / 60
