"""TemporalObject abstract base class."""

import warnings
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generic
from typing import Literal
from typing import TypeVar

import numpy as np
from scipy.integrate import cumulative_trapezoid  # type: ignore
from scipy.integrate import trapezoid

from ..typing_utils import NumpyNumberArray
from ..typing_utils import RealNumber


T = TypeVar("T", bound="TemporalObject[Any]")

WEIGHT_OPTS = Literal["frame_duration"]
INTEGRATION_TYPE_OPTS = Literal["rect", "trapz"]


class TimingError(ValueError):
    """Invalid frame timing."""


class TemporalObject(Generic[T], ABC):
    """TemporalObject abstract base class.

    Attributes:
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing the durations of each frame
    """

    frame_start: NumpyNumberArray
    frame_duration: NumpyNumberArray

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of dataobj."""
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
    def frame_mid(self) -> NumpyNumberArray:
        """Get an array of mid times for each frame."""
        return 0.5 * (self.frame_start + self.frame_end)

    @property
    def frame_end(self) -> NumpyNumberArray:
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

    @property
    def total_duration(self) -> float:
        """Get total scan duration (including any gaps)."""
        return self.end_time - self.start_time

    def has_gaps(self) -> bool:
        """Check if there are any time gaps between frames."""
        return self.total_duration > float(sum(self.frame_duration))

    def get_idx_extract_time(
        self, start_time: RealNumber, end_time: RealNumber
    ) -> tuple[int, int]:
        """Get the start and end indices for extracting a time interval.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, inclusive

        Returns:
            tuple of start index (inclusive) and end index (exclusive) of
            interval to extract

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

    def overlap_with(self, other: T) -> list[tuple[RealNumber, RealNumber]]:
        """Determine temporal overlap with another TemporalObject of same type.

        This is an overlap finding problem in a set of line segments.
        Each frame is a line segment, with start and end of the line segment
        corresponding to frame start and end times, respectively.
        For the purpose of defining overlap, these line segments are treated as
        being closed at the start and open at the end.
        If there is no overlap, the output will be an empty list.

        Args:
            other: TemporalObject to compare to

        Returns:
            list of tuples listing frame start and end times.
        """
        overlap_segs: list[tuple[RealNumber, RealNumber]] = []
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
    def dataobj(self) -> NumpyNumberArray:
        """Get data object, which can be a 2-D or a 3-D matrix.

        The last dimension of dataobj corresponds to time.
        """
        pass

    @abstractmethod
    def extract(self, start_time: RealNumber, end_time: RealNumber) -> T:
        """Extract a time interval.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, inclusive

        Returns:
            extracted time interval
        """
        pass

    @abstractmethod
    def concatenate(self, other: T) -> T:
        """Concatenate with another TemporalObject of same type.

        Args:
            other: TemporalObject to concatenate

        Returns:
            concatenated TemporalObject
        """
        pass

    def split(self, split_time: RealNumber) -> tuple[T, T]:
        """Split into two TemporalObjects, preserving total n of frames.

        Args:
            split_time: time at which to split

        Returns:
            first_img: first of the two split TemporalObjects, not including split_time
            second_img: second of the two split TemporalObjects, including split_time
        """
        first_img = self.extract(self.start_time, split_time)
        second_img = self.extract(split_time, self.end_time)
        return first_img, second_img

    def get_weights(
        self, weight_by: WEIGHT_OPTS | NumpyNumberArray | None = None
    ) -> NumpyNumberArray:
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
        delta: NumpyNumberArray
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

    def _dynamic_mean(
        self,
        weight_by: WEIGHT_OPTS | NumpyNumberArray | None = None,
        integration_type: INTEGRATION_TYPE_OPTS = "rect",
    ) -> NumpyNumberArray:
        """Compute the (weighted) dynamic mean over time.

        Args:
            weight_by: If weight_by == None, each frame is weighted equally.
                       If weight_by == 'frame_duration', each frame is weighted
                       proportionally to its duration (inverse variance weighting).
                       If weight_by is a 1-D array, then specified values are used.
            integration_type: rect (rectangular) or trapz (trapezoidal).

        Returns:
            a 1-D array of weighted temporal averages

        Raises:
            ValueError: invalid integration type
        """
        dyn_mean: NumpyNumberArray
        if integration_type == "rect":
            dyn_mean = np.average(
                self.dataobj,  # type: ignore
                axis=-1,
                weights=self.get_weights(weight_by),
            )
        elif integration_type == "trapz":
            if weight_by:
                warnings.warn(
                    (
                        "When calculating dynamic mean using trapezoidal integration, "
                        "weight_by option is ignored."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            dyn_mean = trapezoid(self.dataobj, self.frame_mid) / (
                self.frame_mid[-1] - self.frame_mid[0]
            )
        else:
            raise ValueError("integration_type" + integration_type + "is invalid")

        return dyn_mean

    def cumulative_integral(
        self, integration_type: INTEGRATION_TYPE_OPTS = "trapz"
    ) -> NumpyNumberArray:
        """Cumulative integration starting at t=0 and ending at each frame_end.

        If start_time > 0, the triangular area between (t=0, 0) and
        (t=start_time, dataobj value at 0th frame) will be added to each
        cumulative integral.

        Args:
            integration_type: rect (rectangular) or trapz (trapezoidal).

        Returns:
            cumulative integral, with same shape as dataobj

        Raises:
            ValueError: invalid integration type
        """
        if self.has_gaps():
            warnings.warn(
                (
                    "TemporalObject has time gaps between frames. "
                    "Make sure that gaps do not mask important temporal dynamics."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            if integration_type == "rect":
                warnings.warn(
                    (
                        "Rectangular integration is not available for "
                        "TemporalObjects with time gaps between frames. "
                        "Changing to trapezoidal integration."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                integration_type = "trapz"

        # if start_time > 0, then we assume activity linearly increased from 0
        # to values attained in first frame
        initial = self.start_time * self.dataobj[..., 0] / 2
        res: NumpyNumberArray

        if integration_type == "rect":
            res = (
                np.cumsum(self.dataobj * self.frame_duration, axis=-1)
                + initial[..., np.newaxis]
            )
        elif integration_type == "trapz":
            res = (
                cumulative_trapezoid(self.dataobj, self.frame_mid, axis=-1, initial=0)
                + initial[..., np.newaxis]
            )
        else:
            raise ValueError("integration_type" + integration_type + "is invalid")
        return res


def check_frametiming(
    frame_start: NumpyNumberArray, frame_duration: NumpyNumberArray
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
