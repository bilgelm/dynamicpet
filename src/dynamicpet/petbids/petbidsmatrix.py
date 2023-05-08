"""PETBIDSMatrix class."""

from copy import deepcopy

from ..temporalobject.temporalmatrix import TemporalMatrix
from ..typing_utils import NumpyRealNumberArray
from ..typing_utils import RealNumber
from .petbidsjson import PetBidsJson
from .petbidsjson import get_frametiming
from .petbidsjson import update_frametiming_from
from .petbidsobject import PETBIDSObject


class PETBIDSMatrix(TemporalMatrix, PETBIDSObject):
    """4D image data with corresponding PET-BIDS time frame information.

    Attributes:
        _dataobj: 1 x num_frames vector or k x num_frames matrix
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing the durations of each frame
        json_dict: PET-BIDS json dictionary
    """

    def __init__(self, dataobj: NumpyRealNumberArray, json_dict: PetBidsJson) -> None:
        """Matrix with corresponding PET-BIDS time frame information.

        Args:
            dataobj: vector or k x num_frames matrix
            json_dict: PET-BIDS json dictionary
        """
        frame_start, frame_duration = get_frametiming(json_dict)

        super().__init__(dataobj, frame_start, frame_duration)

        # need to make a copy of json_dict before storing
        self.json_dict: PetBidsJson = deepcopy(json_dict)

    def extract(self, start_time: RealNumber, end_time: RealNumber) -> "PETBIDSMatrix":
        """Extract a PETBIDSMatrix from a temporally longer PETBIDSMatrix.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, exclusive

        Returns:
            extracted_img: extracted PETBIDSMatrix
        """
        extracted_matrix = super().extract(start_time, end_time)
        json_dict = update_frametiming_from(self.json_dict, extracted_matrix)

        extract_res = PETBIDSMatrix(extracted_matrix.dataobj, json_dict)
        return extract_res

    def concatenate(self, other: "PETBIDSMatrix") -> "PETBIDSMatrix":  # type: ignore
        """Concatenate another PETBIDSMatrix at the end (in time).

        Args:
            other: PETBIDSMatrix to concatenate

        Returns:
            concatenated PETBIDSMatrix

        Raises:
            ValueError: PETBIDSMatrices are from different radionuclides
        """
        if self.json_dict["TracerNuclide"] != other.json_dict["TracerNuclide"]:
            raise ValueError("Cannot concatenate data from different radionuclides")

        concat_mat = super().concatenate(other)
        json_dict = update_frametiming_from(self.json_dict, concat_mat)

        concat_res = PETBIDSMatrix(concat_mat.dataobj, json_dict)

        return concat_res
