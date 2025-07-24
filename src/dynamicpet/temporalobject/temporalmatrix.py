"""TemporalMatrix class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .temporalobject import (
    INTEGRATION_TYPE_OPTS,
    WEIGHT_OPTS,
    TemporalObject,
    TimingError,
    check_frametiming,
)

if TYPE_CHECKING:
    from dynamicpet.typing_utils import NumpyNumberArray, RealNumber


class TemporalMatrix(TemporalObject["TemporalMatrix"]):
    """Matrix with corresponding time frame information.

    Args:
        dataobj: vector or k x num_frames matrix
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing the durations of each frame
        elem_names: [optional] list of k element names

    Attributes:
        _dataobj: 1 x num_frames vector or k x num_frames matrix
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing the durations of each frame
        elem_names: list of k element names

    """

    _dataobj: NumpyNumberArray

    def __init__(
        self,
        dataobj: NumpyNumberArray,
        frame_start: NumpyNumberArray,
        frame_duration: NumpyNumberArray,
        elem_names: list[str] | None = None,
    ) -> None:
        """Matrix with corresponding time frame information.

        Args:
            dataobj: vector or k x num_frames matrix
            frame_start: vector containing the start times of each frame
            frame_duration: vector containing the durations of each frame
            elem_names: [optional] list of k element names

        Raises:
            ValueError: empty dataobj
            TimingError: inconsistent timing info

        """
        check_frametiming(frame_start, frame_duration)
        self.frame_start: NumpyNumberArray = frame_start
        self.frame_duration: NumpyNumberArray = frame_duration

        if dataobj.size == 0:
            msg = "dataobj is empty"
            raise ValueError(msg)

        self._dataobj: NumpyNumberArray
        if dataobj.ndim == 1:
            # if matrix is 1D, store as matrix with a single element in 1st dim
            self._dataobj = dataobj[np.newaxis, :]
        elif dataobj.ndim == 2:  # noqa: PLR2004
            self._dataobj = dataobj
        else:
            msg = "dataobj must be a 1- or 2-D"
            raise ValueError(msg)

        num_frames = len(self.frame_start)
        if self._dataobj.shape[1] != num_frames:
            msg = (
                f"2nd dimension of matrix ({self._dataobj.shape[1]}) must match "
                f"the length of frame timing ({num_frames})"
            )
            raise TimingError(msg)

        k = self._dataobj.shape[0]
        if elem_names is None:
            self.elem_names = [str(i) for i in range(k)]
        elif len(elem_names) == k:
            self.elem_names = elem_names
        else:
            msg = f"length of elem_names must match the number of dataobj rows ({k})"
            raise ValueError(msg)

    @property
    def dataobj(self) -> NumpyNumberArray:
        """Get data object."""
        return self._dataobj

    def get_elem(self, elem: str) -> TemporalMatrix:
        """Get timeseries data for a specific element."""
        i = self.elem_names.index(elem)
        return TemporalMatrix(
            self.dataobj[i],
            self.frame_start,
            self.frame_duration,
            [elem],
        )

    def extract(self, start_time: RealNumber, end_time: RealNumber) -> TemporalMatrix:
        """Extract a temporally shorter TemporalMatrix from a TemporalMatrix.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, inclusive

        Returns:
            extracted TemporalMatrix

        """
        start_index, end_index = self.get_idx_extract_time(start_time, end_time)

        extracted_matrix: NumpyNumberArray = self.dataobj[:, start_index:end_index]

        return TemporalMatrix(
            extracted_matrix,
            self.frame_start[start_index:end_index],
            self.frame_duration[start_index:end_index],
            self.elem_names,
        )

    def dynamic_mean(
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

        """
        return self._dynamic_mean(weight_by, integration_type)

    def concatenate(self, other: TemporalMatrix) -> TemporalMatrix:
        """Concatenate a TemporalMatrix at the end (in time).

        Args:
            other: TemporalMatrix to concatenate

        Returns:
            concatenated TemporalMatrix

        Raises:
            TimingError: TemporalMatrices have temporal overlap or
                         TemporalMatrix being concatenated is earlier in time
            ValueError: TemporalMatrix being concatenated has different element
                        names

        """
        if self.overlap_with(other) != []:
            msg = "Cannot concatenate TemporalMatrices with temporal overlap"
            raise TimingError(msg)
        if self.end_time > other.start_time:
            msg = "TemporalMatrix being concatenated occurs earlier in time"
            raise TimingError(msg)
        if self.elem_names != other.elem_names:
            msg = "TemporalMatrix being concatenated has different element names"
            raise ValueError(msg)

        # concatenation result
        return TemporalMatrix(
            np.concatenate([self.dataobj, other.dataobj], axis=1),
            np.concatenate([self.frame_start, other.frame_start]),
            np.concatenate([self.frame_duration, other.frame_duration]),
            self.elem_names,
        )

    def timeseries_in_mask(
        self,
        mask: NumpyNumberArray | None = None,
    ) -> TemporalMatrix:
        """Get timeseries for each element within a subset of the elements.

        Args:
            mask: Binary mask defining the subset, with shape = (num_elements, 1)

        Returns:
            timeseries (in mask if provided, otherwise a copy of self is returned)

        Raises:
            ValueError: binary mask is incompatible

        """
        if mask is None:
            # we are essentially returning self, but will make a deepcopy
            dataobj = self.dataobj.reshape((self.num_elements, self.num_frames))
        elif mask.shape == self.dataobj.shape[:-1]:
            dataobj = self.dataobj[mask.astype("bool"), :]
        else:
            msg = "Binary mask is incompatible with data"
            raise ValueError(msg)

        # TACs
        return TemporalMatrix(
            dataobj,
            self.frame_start,
            self.frame_duration,
            self.elem_names,
        )
