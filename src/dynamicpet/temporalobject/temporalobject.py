"""TemporalObject abstract base class."""

import warnings
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generic
from typing import List
from typing import Literal
from typing import Tuple
from typing import TypeVar

import numpy as np

from ..typing_utils import NumpyRealNumberArray
from ..typing_utils import RealNumber


T = TypeVar("T", bound="TemporalObject[Any]")

WEIGHT_OPTS = Literal["frame_duration"]


class TimingError(ValueError):
    """Invalid frame timing."""


class TemporalObject(Generic[T], ABC):
    """TemporalObject abstract base class.

    Attributes:
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing the durations of each frame
    """

    frame_start: NumpyRealNumberArray
    frame_duration: NumpyRealNumberArray

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape of data matrix."""
        return self.dataobj.shape

    @property
    def num_elements(self) -> int:
        """Get number of elements in each frame."""
        return int(np.prod(self.dataobj.shape[:-1]))

    @property
    def num_frames(self) -> int:
        """Get number of frames."""
        return len(self.frame_start)

    @property
    def frame_mid(self) -> NumpyRealNumberArray:
        """Get an array of mid times for each frame."""
        return 0.5 * (self.frame_start + self.frame_end)

    @property
    def frame_end(self) -> NumpyRealNumberArray:
        """Get an array of end times for each frame."""
        return self.frame_start + self.frame_duration

    @property
    def start_time(self) -> float:
        """Get the starting time of first frame."""
        return float(self.frame_start[0])

    @property
    def end_time(self) -> float:
        """Get the ending time of last frame."""
        return float(self.frame_end[-1])

    def has_gaps(self) -> bool:
        """Check if there are any time gaps between frames."""
        return self.end_time - self.start_time > float(sum(self.frame_duration))

    def get_idx_extract_time(
        self, start_time: RealNumber, end_time: RealNumber
    ) -> Tuple[int, int]:
        """Get the start and end indices for extracting a time interval.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, exclusive

        Returns:
            start_index: start index of interval to extract, inclusive
            end_index: end index of interval to extract, exclusive

        Raises:
            TimingError: extraction times are out of bound
        """
        if start_time >= end_time:
            raise TimingError("Start time must be before end time")

        if start_time < self.frame_start[0]:
            start_time = self.frame_start[0]
            warnings.warn(
                (
                    "Specified start time is before the start time of "
                    "the first frame. Constraining start time to be the "
                    "start time of the first frame."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        elif start_time > self.frame_end[-1]:
            raise TimingError(
                "Start time is beyond the time covered by the time series data!"
            )

        # find first frame with frame_start at or shortest after start_time
        start_index: int = next(
            (i for i, t in enumerate(self.frame_start) if t >= start_time),
            self.num_frames - 1,
        )

        if end_time > self.frame_end[-1]:
            end_time = self.frame_end[-1]
            warnings.warn(
                (
                    "Specified end time is beyond the end time of the "
                    "last frame. Constraining end time to be the end "
                    "time of the last frame."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        elif end_time < self.frame_start[0]:
            raise TimingError(
                "End time is prior to the time covered by time series data!"
            )

        # find the first time frame with frame_end shortest after the specified end time
        end_index: int = next(
            (i for i, t in enumerate(self.frame_end) if t > end_time),
            self.num_frames,
        )

        # another sanity check, mainly to make sure that start_index!=end_index
        if not start_index < end_index:
            raise TimingError("Start index must be smaller than end index")

        if not self.frame_start[start_index] == start_time:
            warnings.warn(
                (
                    f"Specified start time {start_time} "
                    "did not match the start time of any of the frames. "
                    f"Using {self.frame_start[start_index]} "
                    "as start time instead."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        if not self.frame_end[end_index - 1] == end_time:
            warnings.warn(
                (
                    f"Specified end time {end_time} "
                    "did not match the end time of any of the frames. "
                    f"Using {self.frame_end[end_index - 1]} as end time instead."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        return start_index, end_index

    def overlap_with(self, other: T) -> List[Tuple[RealNumber, RealNumber]]:
        """Determine temporal overlap with another TemporalObject of same type.

        Frame start times are inclusive and end times are exclusive.
        If there is no overlap, the output will be an empty list.

        Args:
            other: TemporalObject to compare to

        Returns:
            list of tuples listing frame start and end times.
        """
        overlap_segs: List[Tuple[RealNumber, RealNumber]] = []
        i = j = 0
        while i < self.num_frames and j < other.num_frames:
            s1 = self.frame_start[i]
            s2 = self.frame_end[i]
            o1 = other.frame_start[j]
            o2 = other.frame_end[j]
            if s2 <= o1:
                # no overlap
                i += 1
            elif s1 >= o2:
                # no overlap
                j += 1
            else:
                u = max(s1, o1)
                v = min(s2, o2)
                overlap_segs.append((u, v))

                if s1 == o1:
                    i += s2 <= o2
                    j += s2 >= o2
                elif s2 == o2:
                    i += s1 > o1
                    j += s1 < o1
                elif s1 < o1:
                    i += s2 < o2
                    j += s2 > o2
                else:
                    i += s2 > o2
                    j += s2 < o2

        return overlap_segs

    @property
    @abstractmethod
    def dataobj(self) -> NumpyRealNumberArray:
        """Get data object, which can be a 2-D or a 3-D matrix."""
        pass

    @abstractmethod
    def extract(self, start_time: RealNumber, end_time: RealNumber) -> T:
        """Extract a time interval.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, exclusive

        Returns:
            extracted time interval
        """
        pass

    @abstractmethod
    def concatenate(self, other: T) -> T:
        """Concatenate with another TemporalObject.

        Args:
            other: TemporalObject to concatenate

        Returns:
            concatenated TemporalObject
        """
        pass

    def split(self, split_time: RealNumber) -> Tuple[T, T]:
        """Split into two TemporalObjects, preserving total n of frames.

        Args:
            split_time: time at which to split

        Returns:
            first_img: first of the two split TemporalObjects, not including split_time
            second_img: second of the two split TemporalObjects, including split_time
        """
        return (
            self.extract(self.start_time, split_time),
            self.extract(split_time, self.end_time),
        )

    def get_weights(
        self, weight_by: WEIGHT_OPTS | NumpyRealNumberArray | None = None
    ) -> NumpyRealNumberArray:
        """Get weights for each time frame.

        Args:
            weight_by: If weight_by == None, each frame is weighted equally.
                       If weight_by == 'frame_duration', each frame is weighted
                       proportionally to its duration (inverse variance weighting).
                       If weight_by is a 1-D array, then specified values are used.

        Returns:
            numeric weights as a vector with num_frames elements

        Raises:
            ValueError: invalid weights
        """
        delta: NumpyRealNumberArray
        if weight_by is None:
            delta = np.ones_like(self.frame_duration)
        elif isinstance(weight_by, str):
            if weight_by == "frame_duration":
                delta = self.frame_duration
            else:
                raise ValueError("{weight_by} is not a valid weights argument")
        elif weight_by.ndim == 1 and len(weight_by) == self.num_frames:
            delta = weight_by
        else:
            raise ValueError("Weights should be None, frame_duration, or a numpy array")
        return delta

    def cumulative_integral(self) -> NumpyRealNumberArray:
        """Cumulative integration starting at t=0 and ending at frame_end.

        Returns:
            cumulative integral

        Raises:
            NotImplementedError: there are time gaps
        """
        if self.has_gaps():
            raise NotImplementedError(
                "Cumulative integration has not been implemented for "
                "TemporalObjects with time gaps between frames"
            )

        # if start_time > 0, then we assume activity linearly increased from 0
        # to values attained in first frame
        initial = self.start_time * self.dataobj[..., 0] / 2
        res: NumpyRealNumberArray = (
            np.cumsum(self.dataobj * self.frame_duration, axis=-1)
            + initial[..., np.newaxis]
        )
        return res


def check_frametiming(
    frame_start: NumpyRealNumberArray, frame_duration: NumpyRealNumberArray
) -> None:
    """Check if frame timing is valid.

    Args:
        frame_start: vector of frame start times
        frame_duration: vector of frame end times

    Raises:
        TimingError: inconsistent timing info
    """
    if not len(frame_start) == len(frame_duration):
        raise TimingError("Unequal number of frame start and end times")

    # in the unusual but possible case where injection occurs after scan start,
    # some FrameTimesStart may negative, so we don't check if start >= 0
    if frame_duration[0] <= 0:
        raise TimingError("Non-positive frame duration at frame 0")
    for i in range(1, len(frame_start)):
        if frame_duration[i] <= 0:
            raise TimingError(f"Non-positive frame duration at frame {i}")
        if frame_start[i - 1] >= frame_start[i]:
            raise TimingError(
                "Non-increasing frame start times: "
                f"{frame_start[i - 1]}, {frame_start[i]}"
            )
        if frame_start[i - 1] + frame_duration[i - 1] > frame_start[i]:
            raise TimingError(f"Previous frame {i - 1} overlaps with current frame {i}")
