"""Test frame timing operations."""

import numpy as np
import pytest
from numpy.typing import NDArray

from dynamicpet.temporalobject import TemporalMatrix


@pytest.fixture
def tm() -> TemporalMatrix:
    """Create a TemporalMatrix."""
    dataobj: NDArray[np.double] = np.array([0, 10, 10, 20, 20, 40, 40], dtype=np.double)
    frame_start: NDArray[np.double] = (
        np.array([0, 5, 10, 20, 30, 40, 50], dtype=np.double) * 60
    )
    frame_duration: NDArray[np.double] = (
        np.array([5, 5, 10, 10, 10, 10, 10], dtype=np.double) * 60
    )

    return TemporalMatrix(dataobj, frame_start, frame_duration)


def test_num_elements(tm: TemporalMatrix) -> None:
    """Test number of frames."""
    assert tm.num_elements == 1


def test_num_frames(tm: TemporalMatrix) -> None:
    """Test number of frames."""
    assert tm.num_frames == 7


def test_start_time(tm: TemporalMatrix) -> None:
    """Test start time."""
    assert tm.start_time == 0


def test_end_time(tm: TemporalMatrix) -> None:
    """Test end time."""
    assert tm.end_time == 3600


def test_frame_duration(tm: TemporalMatrix) -> None:
    """Test frame durations."""
    for i in range(2):
        assert tm.frame_duration[i] == 300
    for i in range(2, 7):
        assert tm.frame_duration[i] == 600


def test_has_gaps_no(tm: TemporalMatrix) -> None:
    """Test if there are gaps between frames when there aren't any."""
    assert not tm.has_gaps()


def test_has_gaps_yes() -> None:
    """Test if there are gaps between frames when there is one."""
    thistm = TemporalMatrix(
        dataobj=np.array([0, 0]),
        frame_start=np.array([0, 10]),
        frame_duration=np.array([5, 5]),
    )
    assert thistm.has_gaps()


def test_extract_time_identity(tm: TemporalMatrix) -> None:
    """Test extract_time with actual start and end time."""
    start_time = tm.start_time
    end_time = tm.end_time
    extr = tm.extract(start_time, end_time)

    assert extr.num_elements == tm.num_elements
    assert extr.num_frames == tm.num_frames
    assert extr.start_time == start_time
    assert extr.end_time == end_time


def test_extract_time_second_half(tm: TemporalMatrix) -> None:
    """Extract second half."""
    frame_start = tm.frame_start

    start_time = frame_start[tm.num_frames // 2]
    end_time = tm.end_time
    extr = tm.extract(start_time, end_time)

    assert extr.num_frames == tm.num_frames - tm.num_frames // 2
    assert extr.start_time == start_time
    assert extr.end_time == end_time


def test_extract_time_first_half(tm: TemporalMatrix) -> None:
    """Extract first half."""
    frame_end = tm.frame_end

    start_time = tm.start_time
    end_time = frame_end[tm.num_frames // 2]
    extr = tm.extract(start_time, end_time)

    assert extr.num_frames == tm.num_frames // 2 + 1
    assert extr.start_time == start_time
    assert extr.end_time == end_time


def test_extract_time_middle(tm: TemporalMatrix) -> None:
    """Extract the middle portion."""
    frame_start = tm.frame_start
    frame_end = tm.frame_end

    start_time = frame_start[1]
    end_time = frame_end[-2]
    extr = tm.extract(start_time, end_time)

    assert extr.num_frames == tm.num_frames - 2
    assert extr.start_time == start_time
    assert extr.end_time == end_time


def test_extract_time_middle_fuzzy(tm: TemporalMatrix) -> None:
    """Extract the middle portion given timings that do not exactly match."""
    frame_start = tm.frame_start
    frame_end = tm.frame_end

    start_time = frame_start[1] + 6
    end_time = frame_end[-2] - 6
    extr = tm.extract(start_time, end_time)

    assert extr.num_frames == tm.num_frames - 4
    assert extr.start_time == frame_start[2]
    assert extr.end_time == frame_end[-3]


def test_overlap_with_self(tm: TemporalMatrix) -> None:
    """Test overlap with self."""
    expected_result = [
        (tm.frame_start[i], tm.frame_end[i]) for i in range(0, tm.num_frames)
    ]
    assert tm.overlap_with(tm) == expected_result


def test_overlap_with_nonoverlapping(tm: TemporalMatrix) -> None:
    """Test overlap with a nonoverlapping TemporalMatrix."""
    other = TemporalMatrix(
        dataobj=np.array([0, 0]),
        frame_start=np.array([tm.end_time, tm.end_time + 100], dtype=float),
        frame_duration=np.array([100, 100]),
    )
    assert tm.overlap_with(other) == []


def test_overlap_with_one_frame_not_subset(tm: TemporalMatrix) -> None:
    """Test overlap with a 1-frame overlapping FC that is not a subset."""
    other = TemporalMatrix(
        dataobj=np.array([0, 0]),
        frame_start=np.array([tm.frame_start[-1], tm.end_time], dtype=float),
        frame_duration=np.array([tm.frame_duration[-1]] * 2),
    )
    assert tm.overlap_with(other) == [(tm.frame_start[-1], tm.end_time)]
    assert other.overlap_with(tm) == [(tm.frame_start[-1], tm.end_time)]


def test_overlap_with_subset(tm: TemporalMatrix) -> None:
    """Test overlap with a multiframe overlapping FC that is a subset."""
    other = TemporalMatrix(
        dataobj=tm.dataobj[:, 1:-1],
        frame_start=tm.frame_start[1:-1],
        frame_duration=tm.frame_duration[1:-1],
    )
    expected_result = [
        (tm.frame_start[i], tm.frame_end[i]) for i in range(1, tm.num_frames - 1)
    ]
    assert tm.overlap_with(other) == expected_result
    assert other.overlap_with(tm) == expected_result


def test_overlap_two_frames_within_one(tm: TemporalMatrix) -> None:
    """Test overlap with two frames within one in the other."""
    mid_frame = (tm.start_time + tm.frame_start[1]) / 2
    other = TemporalMatrix(
        dataobj=np.array([0, 0]),
        frame_start=np.array([tm.start_time, mid_frame]),
        frame_duration=np.array([tm.frame_duration[0] / 2] * 2),
    )
    assert tm.overlap_with(other) == [
        (tm.start_time, mid_frame),
        (mid_frame, tm.frame_end[0]),
    ]


def test_concatenate(tm: TemporalMatrix) -> None:
    """Test concatenation."""
    dataobj: NDArray[np.double] = np.array([0, 10, 10, 20, 20, 40, 40], dtype=np.double)
    frame_start: NDArray[np.double] = (
        np.array([60, 65, 70, 80, 90, 100, 110], dtype=np.double) * 60
    )
    frame_duration: NDArray[np.double] = (
        np.array([5, 5, 10, 10, 10, 10, 10], dtype=np.double) * 60
    )
    tm2 = TemporalMatrix(dataobj, frame_start, frame_duration)
    tm_concat = tm.concatenate(tm2)

    assert tm_concat.num_frames == tm.num_frames + tm2.num_frames


def test_dynamic_mean_unweighted(tm: TemporalMatrix) -> None:
    """Test unweighted dynamic mean."""
    assert tm.dynamic_mean() == tm.dataobj.mean()


def test_dynamic_mean_frame_duration_weighted(tm: TemporalMatrix) -> None:
    """Test frame duration weighted dynamic mean."""
    expected_result = (tm.dataobj * tm.frame_duration).sum() / tm.frame_duration.sum()
    assert tm.dynamic_mean(weight_by="frame_duration") == expected_result


def test_dynamic_mean_custom_weighted(tm: TemporalMatrix) -> None:
    """Test custom weighted dynamic mean."""
    weights = np.zeros((tm.num_frames,))
    weights[-1] = 10
    assert tm.dynamic_mean(weight_by=weights) == tm.dataobj[:, -1]


def test_dynamic_mean_trapz(tm: TemporalMatrix) -> None:
    """Test trapezoidal integration based dynamic mean."""
    assert np.allclose(tm.dynamic_mean(integration_type="trapz"),
                       tm.dynamic_mean(weight_by="frame_duration"),
                       rtol=.03)


def test_cumulative_integral(tm: TemporalMatrix) -> None:
    """Test cumulative integral."""
    res = tm.cumulative_integral(integration_type="rect")
    expected = np.array([0, 50, 150, 350, 550, 950, 1350]) * 60
    assert res.shape == (tm.num_elements, tm.num_frames)
    assert np.all(expected == res)


def test_cumulative_integral_trapz(tm: TemporalMatrix) -> None:
    """Test cumulative trapezoidal integral based on frame mid times."""
    res = tm.cumulative_integral("trapz")
    expected = np.array([0, 1500, 6000, 15000, 27000, 45000, 69000])
    assert res.shape == (tm.num_elements, tm.num_frames)
    assert np.all(expected == res)


def test_cumulative_integral_broadcasting() -> None:
    """Test cumulative integral with potentially incorrect axis broadcasting."""
    dataobj: NDArray[np.double] = np.array([[0, 10], [0, 20]], dtype=np.double)
    frame_start: NDArray[np.double] = np.array([0, 5], dtype=np.double)
    frame_duration: NDArray[np.double] = np.array([5, 5], dtype=np.double)

    tm2 = TemporalMatrix(dataobj, frame_start, frame_duration)
    res = tm2.cumulative_integral(integration_type="rect")
    expected = np.array([[0, 50], [0, 100]])
    assert res.shape == (tm2.num_elements, tm2.num_frames)
    assert np.all(expected == res)
