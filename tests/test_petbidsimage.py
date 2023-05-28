"""Test cases for the PETBIDSImage class."""
from typing import Callable
from typing import Generator

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray
from pytest_mock import MockerFixture

from dynamicpet.petbids import PETBIDSImage
from dynamicpet.petbids import load
from dynamicpet.petbids.petbidsjson import PetBidsJson
from dynamicpet.petbids.petbidsjson import get_frametiming


@pytest.fixture
def json_dict() -> PetBidsJson:
    """Create a minimal PET-BIDS json dictionary for testing purposes.

    Returns:
        PET-BIDS json dictionary
    """
    frame_start: NDArray[np.int16] = (
        np.array([0, 5, 10, 20, 30, 40, 50], dtype=np.int16) * 60
    )
    frame_end: NDArray[np.int16] = (
        np.array([5, 10, 20, 30, 40, 50, 60], dtype=np.int16) * 60
    )
    frame_duration: NDArray[np.int16] = frame_end - frame_start

    json_dict: PetBidsJson = {
        "FrameTimesStart": frame_start.tolist(),
        "FrameDuration": frame_duration.tolist(),
        "InjectionStart": 0,
        "ScanStart": 0,
        "TracerNuclide": "C11",
    }

    return json_dict


@pytest.fixture
def img(json_dict: PetBidsJson) -> Nifti1Image:
    """Simulate a 4D dynamic PET image.

    Returns:
       Simulated 4D dynamic PET image
    """
    dims = (10, 11, 12, 7)
    img_dat = np.zeros(dims)

    r1 = 1.0
    dvr = 1.2
    k2 = 1.1  # 1/minute

    frame_start, frame_duration = get_frametiming(json_dict)
    frame_end = frame_start + frame_duration

    c_ref = np.array([0, 100, 200, 160, 140, 120, 120], dtype=np.float64)
    t = 0.5 * (frame_start + frame_end)
    c_t: NDArray[np.generic] = r1 * c_ref + np.convolve(
        (k2 - r1 * k2 / dvr) * c_ref, np.exp(-k2 * t / dvr), "same"
    )

    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if k < (dims[2] // 2):
                    img_dat[i, j, k, :] = c_ref
                else:
                    img_dat[i, j, k, :] = c_t

    # save 4D image
    img = Nifti1Image(img_dat, np.eye(4))  # type: ignore

    return img


@pytest.fixture
def ti(img: Nifti1Image, json_dict: PetBidsJson) -> PETBIDSImage:
    """Create a PETBIDSImage object for testing purposes."""
    return PETBIDSImage(img, json_dict)


def test_load(
    img: Nifti1Image,
    json_dict: PetBidsJson,
    mocker: Callable[..., Generator[MockerFixture, None, None]],
) -> None:
    """Test load from mocked files."""
    mocker.patch(  # type: ignore
        "dynamicpet.petbids.petbidsimage.nib_load", return_value=img
    )
    mocker.patch(  # type: ignore
        "dynamicpet.petbids.petbidsimage.read_json", return_value=json_dict
    )
    mocker.patch("dynamicpet.petbids.petbidsimage.op.exists", return_value=True)  # type: ignore
    load("mock_input.nii.gz", "mock_input.json")


def test_pointer(img: Nifti1Image, json_dict: PetBidsJson) -> None:
    """Test a possible json_dict pointer problem for PETBIDSImage instance.

    Test if manipulating json_dict used to create a PETBIDSImage instance
    alters the attributes of the instance.
    """
    myti = PETBIDSImage(img, json_dict)
    json_dict["FrameTimesStart"][0] = 9999
    assert myti.start_time == 0
    assert myti.start_time == myti.json_dict["FrameTimesStart"][0]


def test_num_voxels(ti: PETBIDSImage, img: Nifti1Image) -> None:
    """Test number of voxels."""
    assert ti.num_voxels == img.dataobj[..., 0].size


def test_num_frames(ti: PETBIDSImage, img: Nifti1Image) -> None:
    """Test number of frames."""
    assert ti.num_frames == img.shape[3]
    assert ti.num_frames == 7


def test_start_time(ti: PETBIDSImage, json_dict: PetBidsJson) -> None:
    """Test start time."""
    if json_dict["InjectionStart"] == 0:
        assert ti.start_time == json_dict["FrameTimesStart"][0] / 60
    elif json_dict["ScanStart"] == 0:
        assert (
            ti.start_time
            == (json_dict["FrameTimesStart"][0] - json_dict["InjectionStart"]) / 60
        )


def test_end_time(ti: PETBIDSImage, json_dict: PetBidsJson) -> None:
    """Test end time."""
    if json_dict["InjectionStart"] == 0:
        assert (
            ti.end_time
            == (json_dict["FrameTimesStart"][-1] + json_dict["FrameDuration"][-1]) / 60
        )
    elif json_dict["ScanStart"] == 0:
        assert (
            ti.end_time
            == (
                json_dict["FrameTimesStart"][-1]
                + json_dict["FrameDuration"][-1]
                - json_dict["InjectionStart"]
            )
            / 60
        )


def test_frame_duration(ti: PETBIDSImage, json_dict: PetBidsJson) -> None:
    """Test frame durations."""
    for i, duration in enumerate(json_dict["FrameDuration"]):
        assert ti.frame_duration[i] == duration / 60


def test_extract_time_identity(ti: PETBIDSImage) -> None:
    """Test extract_time with actual start and end time."""
    start_time = ti.start_time
    end_time = ti.end_time
    extr = ti.extract(start_time, end_time)

    assert extr.shape == ti.shape
    assert extr.start_time == start_time
    assert extr.end_time == end_time


def test_extract_time_second_half(ti: PETBIDSImage) -> None:
    """Extract second half."""
    frame_start = ti.frame_start

    start_time = frame_start[ti.num_frames // 2]
    end_time = ti.end_time
    extr = ti.extract(start_time, end_time)

    assert extr.num_frames == ti.num_frames - ti.num_frames // 2
    assert extr.start_time == start_time
    assert extr.end_time == end_time


def test_extract_time_first_half(ti: PETBIDSImage) -> None:
    """Extract first half."""
    frame_end = ti.frame_end

    start_time = ti.start_time
    end_time = frame_end[ti.num_frames // 2]
    extr = ti.extract(start_time, end_time)

    assert extr.num_frames == ti.num_frames // 2 + 1
    assert extr.start_time == start_time
    assert extr.end_time == end_time


def test_extract_time_middle(ti: PETBIDSImage) -> None:
    """Extract the middle portion."""
    frame_start = ti.frame_start
    frame_end = ti.frame_end

    start_time = frame_start[1]
    end_time = frame_end[-2]
    extr = ti.extract(start_time, end_time)

    assert extr.num_frames == ti.num_frames - 2
    assert extr.start_time == start_time
    assert extr.end_time == end_time


def test_extract_time_middle_fuzzy(ti: PETBIDSImage) -> None:
    """Extract the middle portion given timings that do not exactly match."""
    frame_start = ti.frame_start
    frame_end = ti.frame_end

    start_time = frame_start[1] + 0.1
    end_time = frame_end[-2] - 0.1
    extr = ti.extract(start_time, end_time)

    assert extr.num_frames == ti.num_frames - 4
    assert extr.start_time == frame_start[2]
    assert extr.end_time == frame_end[-3]


def test_split_first(ti: PETBIDSImage) -> None:
    """Split after first frame."""
    split_time = ti.frame_start[1]
    first_img, second_img = ti.split(split_time)

    assert first_img.num_frames == 1
    assert first_img.start_time == ti.start_time
    assert first_img.end_time == split_time

    assert second_img.num_frames == ti.num_frames - 1
    assert second_img.start_time == split_time
    assert second_img.end_time == ti.end_time


def test_split_time_last(ti: PETBIDSImage) -> None:
    """Split before last frame."""
    split_time = ti.frame_start[-1]
    first_img, second_img = ti.split(split_time)

    assert first_img.num_frames == ti.num_frames - 1
    assert first_img.start_time == ti.start_time
    assert first_img.end_time == split_time

    assert second_img.num_frames == 1
    assert second_img.start_time == split_time
    assert second_img.end_time == ti.end_time


def test_split_time_middle(ti: PETBIDSImage) -> None:
    """Split in the middle."""
    split_time = ti.frame_start[1:2].mean()
    first_img, second_img = ti.split(split_time)

    assert first_img.num_frames == 1
    assert first_img.start_time == ti.start_time
    assert first_img.end_time == split_time

    assert second_img.num_frames == ti.num_frames - 1
    assert second_img.start_time == split_time
    assert second_img.end_time == ti.end_time


def test_overlap_with_self(ti: PETBIDSImage) -> None:
    """Test overlap with self."""
    expected_result = [
        (ti.frame_start[i], ti.frame_end[i]) for i in range(0, ti.num_frames)
    ]
    assert ti.overlap_with(ti) == expected_result


def test_decay_correction_factor(ti: PETBIDSImage) -> None:
    """Test decay correction factor."""
    assert ti.get_decay_correction_factor().shape == ti.frame_duration.shape


def test_decay_correct_uncorrect(ti: PETBIDSImage) -> None:
    """Test if decay correction then uncorrection yields same result."""
    ti2 = ti.decay_correct().decay_uncorrect()
    assert np.allclose(ti.dataobj, ti2.dataobj)
    assert np.all(ti.frame_start == ti2.frame_start)
    assert np.all(ti.frame_end == ti2.frame_end)


def test_decay_uncorrect_correct(ti: PETBIDSImage) -> None:
    """Test if decay uncorrection then correction yields same result."""
    ti2 = ti.decay_uncorrect().decay_correct()
    assert np.allclose(ti.dataobj, ti2.dataobj)
    assert np.all(ti.frame_start == ti2.frame_start)
    assert np.all(ti.frame_end == ti2.frame_end)
