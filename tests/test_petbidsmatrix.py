"""Test cases for the PETBIDSMatrix class."""

# ruff: noqa: S101

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from dynamicpet.petbids import PETBIDSMatrix
from dynamicpet.petbids.petbidsmatrix import load

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dynamicpet.petbids.petbidsjson import PetBidsJson


@pytest.fixture
def pm() -> PETBIDSMatrix:
    """Create a TemporalMatrix."""
    dataobj: NDArray[np.double] = np.array([0, 10, 10, 20, 20, 40, 40], dtype=np.double)
    frame_start: NDArray[np.int16] = (
        np.array([0, 5, 10, 20, 30, 40, 50], dtype=np.int16) * 60
    )
    frame_end: NDArray[np.int16] = (
        np.array([5, 10, 20, 30, 40, 50, 60], dtype=np.int16) * 60
    )
    frame_duration: NDArray[np.int16] = frame_end - frame_start

    json_dict: PetBidsJson = {
        "TimeZero": "10:00:00",
        "FrameTimesStart": frame_start.tolist(),
        "FrameDuration": frame_duration.tolist(),
        "InjectionStart": 0,
        "ScanStart": 0,
        "TracerRadionuclide": "C11",
        "ImageDecayCorrected": True,
        "ImageDecayCorrectionTime": 0,
    }

    return PETBIDSMatrix(dataobj, json_dict)


def test_extract(pm: PETBIDSMatrix) -> None:
    """Test extracting couple frames in the middle."""
    start_time = 10
    end_time = 30
    extract_res = pm.extract(start_time, end_time)
    correct_num_frames = 2

    assert extract_res.num_frames == correct_num_frames, (
        f"Expected {correct_num_frames} frames, got {extract_res.num_frames}"
    )
    assert extract_res.num_elements == pm.num_elements, (
        f"Expected {pm.num_elements} elements, got {extract_res.num_elements}"
    )
    assert extract_res.start_time == start_time, (
        f"Expected {start_time} start time, got {extract_res.start_time}"
    )
    assert extract_res.end_time == end_time, (
        f"Expected {end_time} end time, got {extract_res.end_time}"
    )


def test_file_io(pm: PETBIDSMatrix, tmp_path: Path) -> None:
    """Test writing to file and reading it back."""
    fname = tmp_path / "test.tsv"
    pm.to_filename(fname, save_json=True)
    pm2 = load(fname)

    assert np.allclose(pm.frame_start, pm2.frame_start), "Mismatch in frame starts"
    assert np.allclose(pm.frame_duration, pm2.frame_duration), (
        "Mismatch in frame durations"
    )
    assert pm.elem_names == pm2.elem_names, "Mismatch in element names"
    assert np.allclose(pm.dataobj, pm2.dataobj), "Mismatch in dataobj"


def test_decay_correct0_corrected(pm: PETBIDSMatrix) -> None:
    """Test if decay correction on an already corrected TACs does nothing."""
    pm2 = pm.decay_correct()
    assert np.allclose(pm.dataobj, pm2.dataobj), "Mismatch in dataobj"
    assert np.all(pm.frame_start == pm2.frame_start), "Mismatch in frame starts"
    assert np.all(pm.frame_end == pm2.frame_end), "Mismatch in frame ends"


def test_decay_correct_corrected(pm: PETBIDSMatrix) -> None:
    """Test if decay correct-uncorrect-correct yields same result."""
    pm2 = pm.decay_correct(-100)
    assert not np.allclose(pm.dataobj, pm2.dataobj), "Dataobj should be different"
    assert np.all(pm.frame_start == pm2.frame_start), "Mismatch in frame starts"
    assert np.all(pm.frame_end == pm2.frame_end), "Mismatch in frame ends"


def test_decay_correct_uncorrect_correct(pm: PETBIDSMatrix) -> None:
    """Test if decay correct-uncorrect-correct yields same result."""
    pm2 = pm.decay_correct(-100).decay_uncorrect().decay_correct(0)
    assert np.allclose(pm.dataobj, pm2.dataobj), "Mismatch in dataobj"
    assert np.all(pm.frame_start == pm2.frame_start), "Mismatch in frame starts"
    assert np.all(pm.frame_end == pm2.frame_end), "Mistmatch in frame ends"


def test_decay_uncorrect_correct(pm: PETBIDSMatrix) -> None:
    """Test if decay uncorrection then correction yields same result."""
    pm2 = pm.decay_uncorrect().decay_correct()
    assert np.allclose(pm.dataobj, pm2.dataobj), "Mismatch in dataobj"
    assert np.all(pm.frame_start == pm2.frame_start), "Mismatch in frame starts"
    assert np.all(pm.frame_end == pm2.frame_end), "Mismatch in frame ends"


def test_set_timezero(pm: PETBIDSMatrix) -> None:
    """Test setting time zero to InjectionStart then back to ScanStart."""
    zero = 0
    scan_to_inj = 3600  # in seconds
    s_to_min = 1.0 / 60

    pm.json_dict["InjectionStart"] = -scan_to_inj

    pm.set_timezero("InjectionStart")
    assert pm.json_dict["InjectionStart"] == zero, (
        f"Expected {zero} InjectionStart, got {pm.json_dict['InjectionStart']}"
    )
    assert pm.json_dict["ScanStart"] == scan_to_inj, (
        f"Expected {scan_to_inj} ScanStart, got {pm.json_dict['ScanStart']}"
    )
    assert pm.frame_start[0] == scan_to_inj * s_to_min, (
        f"Expected {scan_to_inj * s_to_min} frame start, got {pm.frame_start[0]}"
    )

    pm.set_timezero("ScanStart")
    assert pm.json_dict["ScanStart"] == zero, (
        f"Expected {zero} ScanStart, got {pm.json_dict['ScanStart']}"
    )
    assert pm.json_dict["InjectionStart"] == -scan_to_inj, (
        f"Expected {-scan_to_inj} InjectionStart, got {pm.json_dict['InjectionStart']}"
    )
    assert pm.frame_start[0] == zero, (
        f"Expected {zero} frame start, got {pm.frame_start[0]}"
    )
