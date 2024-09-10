"""Test cases for the PETBIDSMatrix class."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dynamicpet.petbids import PETBIDSMatrix
from dynamicpet.petbids.petbidsjson import PetBidsJson
from dynamicpet.petbids.petbidsmatrix import load


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

    assert extract_res.num_frames == 2
    assert extract_res.num_elements == pm.num_elements
    assert extract_res.start_time == start_time
    assert extract_res.end_time == end_time


def test_file_io(pm: PETBIDSMatrix, tmp_path: Path) -> None:
    """Test writing to file and reading it back."""
    fname = tmp_path / "test.tsv"
    pm.to_filename(fname)
    pm2 = load(fname)

    assert np.allclose(pm.frame_start, pm2.frame_start)
    assert np.allclose(pm.frame_duration, pm.frame_duration)
    assert pm.elem_names == pm2.elem_names
    assert np.allclose(pm.dataobj, pm2.dataobj)


def test_decay_correct0_corrected(pm: PETBIDSMatrix) -> None:
    """Test if decay correction on an already corrected TACs does nothing."""
    pm2 = pm.decay_correct()
    assert np.allclose(pm.dataobj, pm2.dataobj)
    assert np.all(pm.frame_start == pm2.frame_start)
    assert np.all(pm.frame_end == pm2.frame_end)


def test_decay_correct_corrected(pm: PETBIDSMatrix) -> None:
    """Test if decay correct-uncorrect-correct yields same result."""
    pm2 = pm.decay_correct(-100)
    assert not np.allclose(pm.dataobj, pm2.dataobj)
    assert np.all(pm.frame_start == pm2.frame_start)
    assert np.all(pm.frame_end == pm2.frame_end)


def test_decay_correct_uncorrect_correct(pm: PETBIDSMatrix) -> None:
    """Test if decay correct-uncorrect-correct yields same result."""
    pm2 = pm.decay_correct(-100).decay_uncorrect().decay_correct(0)
    assert np.allclose(pm.dataobj, pm2.dataobj)
    assert np.all(pm.frame_start == pm2.frame_start)
    assert np.all(pm.frame_end == pm2.frame_end)


def test_decay_uncorrect_correct(pm: PETBIDSMatrix) -> None:
    """Test if decay uncorrection then correction yields same result."""
    pm2 = pm.decay_uncorrect().decay_correct()
    assert np.allclose(pm.dataobj, pm2.dataobj)
    assert np.all(pm.frame_start == pm2.frame_start)
    assert np.all(pm.frame_end == pm2.frame_end)
