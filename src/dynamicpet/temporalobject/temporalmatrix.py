"""TemporalMatrix class."""

import numpy as np

from ..typing_utils import NumpyRealNumberArray
from ..typing_utils import RealNumber
from .temporalobject import WEIGHT_OPTS
from .temporalobject import TemporalObject
from .temporalobject import TimingError
from .temporalobject import check_frametiming


class TemporalMatrix(TemporalObject["TemporalMatrix"]):
    """Matrix with corresponding time frame information.

    Args:
        dataobj: vector or k x num_frames matrix
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing the durations of each frame

    Attributes:
        _dataobj: 1 x num_frames vector or k x num_frames matrix
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing the durations of each frame
    """

    _dataobj: NumpyRealNumberArray

    def __init__(
        self,
        dataobj: NumpyRealNumberArray,
        frame_start: NumpyRealNumberArray,
        frame_duration: NumpyRealNumberArray,
    ) -> None:
        """Matrix with corresponding time frame information.

        Args:
            dataobj: vector or k x num_frames matrix
            frame_start: vector containing the start times of each frame
            frame_duration: vector containing the durations of each frame

        Raises:
            ValueError: empty dataobj
            TimingError: inconsistent timing info
        """
        check_frametiming(frame_start, frame_duration)
        self.frame_start: NumpyRealNumberArray = frame_start
        self.frame_duration: NumpyRealNumberArray = frame_duration

        if dataobj.size == 0:
            raise ValueError("dataobj is empty")

        self._dataobj: NumpyRealNumberArray
        if dataobj.ndim == 1:
            # if matrix is 1D, store as matrix with a single element in 1st dim
            self._dataobj = dataobj[np.newaxis, :]
        elif dataobj.ndim == 2:
            self._dataobj = dataobj
        else:
            raise ValueError("dataobj must be a 1- or 2-D")

        if not self._dataobj.shape[1] == len(self.frame_start):
            raise TimingError(
                f"2nd dimension of matrix ({self._dataobj.shape[1]}) must "
                f"match the number of columns ({len(self.frame_start)}) in "
                "frame timing information"
            )

    @property
    def dataobj(self) -> NumpyRealNumberArray:
        """Get data object."""
        return self._dataobj

    def extract(self, start_time: RealNumber, end_time: RealNumber) -> "TemporalMatrix":
        """Extract a temporally shorter TemporalMatrix from a TemporalMatrix.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, inclusive

        Returns:
            extracted TemporalMatrix
        """
        start_index, end_index = self.get_idx_extract_time(start_time, end_time)

        extracted_matrix: NumpyRealNumberArray = self.dataobj[:, start_index:end_index]

        return TemporalMatrix(
            extracted_matrix,
            self.frame_start[start_index:end_index],
            self.frame_duration[start_index:end_index],
        )

    def dynamic_mean(
        self,
        weight_by: WEIGHT_OPTS | NumpyRealNumberArray | None = None,
    ) -> NumpyRealNumberArray:
        """Compute the (weighted) dynamic mean over time.

        Args:
            weight_by: If weight_by == None, each frame is weighted equally.
                       If weight_by == 'frame_duration', each frame is weighted
                       proportionally to its duration (inverse variance weighting).
                       If weight_by is a 1-D array, then specified values are used.

        Returns:
            a 1-D array of weighted temporal averages
        """
        dyn_mean: NumpyRealNumberArray = np.average(
            self.dataobj, axis=-1, weights=self.get_weights(weight_by)
        )

        return dyn_mean

    def concatenate(self, other: "TemporalMatrix") -> "TemporalMatrix":
        """Concatenate a TemporalMatrix at the end (in time).

        Args:
            other: TemporalMatrix to concatenate

        Returns:
            concatenated TemporalMatrix

        Raises:
            TimingError: TemporalMatrices have temporal overlap or
                         TemporalMatrix being concatenated is earlier in time
        """
        if self.overlap_with(other) != []:
            raise TimingError(
                "Cannot concatenate TemporalMatrices with temporal overlap"
            )
        if self.end_time > other.start_time:
            raise TimingError(
                "TemporalMatrix being concatenated occurs earlier in time"
            )

        concat_res = TemporalMatrix(
            np.concatenate([self.dataobj, other.dataobj], axis=1),
            np.concatenate([self.frame_start, other.frame_start]),
            np.concatenate([self.frame_duration, other.frame_duration]),
        )

        return concat_res

    def timeseries_in_mask(
        self, mask: NumpyRealNumberArray | None = None
    ) -> "TemporalMatrix":
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
            raise ValueError("Binary mask is incompatible with data")

        tacs = TemporalMatrix(
            dataobj,
            self.frame_start,
            self.frame_duration,
        )
        return tacs
