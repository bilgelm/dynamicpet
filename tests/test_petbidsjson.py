"""Test cases for the petbidsjson file."""

import numpy as np
import pytest

from dynamicpet.petbids.petbidsjson import PetBidsJson
from dynamicpet.petbids.petbidsjson import get_frametiming
from dynamicpet.petbids.petbidsjson import get_radionuclide_halflife


def test_get_frametiming_from_json_scanstart0() -> None:
    """Test getting frame_start and frame_end from json when scan start is 0."""
    my_json_dict: PetBidsJson = {
        "InjectionStart": -120,
        "ScanStart": 0,
        "FrameTimesStart": [0, 120, 240],
        "FrameDuration": [120, 120, 120],
        "TracerNuclide": "C11",
    }
    frame_start, frame_duration = get_frametiming(my_json_dict)
    assert np.all(frame_start == np.array([120, 240, 360]) / 60)
    assert np.all(frame_duration == np.array([120, 120, 120]) / 60)


def test_get_frametiming_from_invalid_jsons() -> None:
    """Test getting frame_start and frame_end from invalid json."""
    my_json_dict: PetBidsJson = {
        "InjectionStart": 1,
        "ScanStart": 1,
        "FrameTimesStart": [0, 120, 240],
        "FrameDuration": [120, 120, 120],
        "TracerNuclide": "C11",
    }

    with pytest.raises(ValueError) as excinfo:
        get_frametiming(my_json_dict)
    assert str(excinfo.value) == "Neither InjectionStart nor ScanStart is 0"
    # my_json_dict: PetBidsJson = {
    #     "InjectionStart": 0,
    #     "ScanStart": 0,
    #     "FrameTimesStart": [999, 120, 240],
    #     "FrameDuration": [120, 120, 120],
    #     "TracerNuclide": "C11",
    # }
    # with pytest.raises(TimingError) as excinfo:
    #     get_frametiming(my_json_dict)
    # assert "Non-increasing frame start times" in str(excinfo.value)


def test_halflife() -> None:
    """Test halflife for C11."""
    my_json_dict: PetBidsJson = {
        "InjectionStart": -120,
        "ScanStart": 0,
        "FrameTimesStart": [0, 120, 240],
        "FrameDuration": [120, 120, 120],
        "TracerNuclide": "C11",
    }
    assert get_radionuclide_halflife(my_json_dict) == 1224 / 60
