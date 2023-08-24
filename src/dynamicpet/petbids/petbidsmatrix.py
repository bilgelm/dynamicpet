"""PETBIDSMatrix class."""

import csv
import os.path as op
from copy import deepcopy
from os import PathLike

import numpy as np

from ..temporalobject.temporalmatrix import TemporalMatrix
from ..typing_utils import NumpyRealNumberArray
from ..typing_utils import RealNumber
from .petbidsjson import PetBidsJson
from .petbidsjson import get_frametiming_in_mins
from .petbidsjson import read_json
from .petbidsjson import update_frametiming_from
from .petbidsjson import write_json
from .petbidsobject import PETBIDSObject


class PETBIDSMatrix(TemporalMatrix, PETBIDSObject):
    """4-D image data with corresponding PET-BIDS time frame information.

    Args:
        dataobj: vector or k x num_frames matrix
        json_dict: PET-BIDS json dictionary
        elem_names: list of k ROI names

    Attributes:
        _dataobj: 1 x num_frames vector or k x num_frames matrix
        frame_start: vector containing the start times of each frame, in min
        frame_duration: vector containing the durations of each frame, in min
        json_dict: PET-BIDS json dictionary
        elem_names: list of k element names
    """

    def __init__(
        self,
        dataobj: NumpyRealNumberArray,
        json_dict: PetBidsJson,
        elem_names: list[str] | None = None,
    ) -> None:
        """Matrix with corresponding PET-BIDS time frame information.

        Args:
            dataobj: vector or k x num_frames matrix
            json_dict: PET-BIDS json dictionary
            elem_names: list of k ROI names
        """
        frame_start, frame_duration = get_frametiming_in_mins(json_dict)

        super().__init__(dataobj, frame_start, frame_duration, elem_names)

        # need to make a copy of json_dict before storing
        self.json_dict: PetBidsJson = deepcopy(json_dict)

    # def get_elem(self, elem: str) -> "PETBIDSMatrix":
    #     """Get timeseries data for a specific element."""
    #     i = self.elem_names.index(elem)
    #     return PETBIDSMatrix(self.dataobj[i], self.json_dict, [elem])

    def extract(self, start_time: RealNumber, end_time: RealNumber) -> "PETBIDSMatrix":
        """Extract a temporally shorter PETBIDSMatrix from a PETBIDSMatrix.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, inclusive

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
        if (
            self.json_dict["TracerRadionuclide"]
            != other.json_dict["TracerRadionuclide"]
        ):
            raise ValueError("Cannot concatenate data from different radionuclides")

        concat_mat = super().concatenate(other)
        json_dict = update_frametiming_from(self.json_dict, concat_mat)

        concat_res = PETBIDSMatrix(concat_mat.dataobj, json_dict)

        return concat_res

    def to_filename(self, filename: str | PathLike[str]) -> None:
        """Save to file.

        Args:
            filename: file name for the tabular TAC tsv output

        Raises:
            ValueError: file is not a tsv file
        """
        fbase, fext = op.splitext(filename)
        if fext != ".tsv":
            raise ValueError("output file must be a tsv file")
        jsonfilename = fbase + ".json"

        with open(filename, "w") as f:
            tsvwriter = csv.writer(f, delimiter="\t")
            tsvwriter.writerow(self.elem_names)
            for row in self.dataobj.T:
                tsvwriter.writerow(row)

        write_json(self.json_dict, jsonfilename)


def load(
    filename: str | PathLike[str], jsonfilename: str | PathLike[str] | None = None
) -> PETBIDSMatrix:
    """Read a tsv file containing temporal entries.

    Each column of the tsv file should be a time activity curve.

    Args:
        filename: path to the tsv file
        jsonfilename: path to the PET BIDS json file

    Returns:
        PETBIDSMatrix created from tsv file

    Raises:
        ValueError: file is not a tsv file
    """
    fbase, fext = op.splitext(filename)
    if fext != ".tsv":
        raise ValueError("output file must be a tsv file")

    if jsonfilename is None:
        jsonfilename = fbase + ".json"

    tsv = []
    with open(filename) as f:
        tsvreader = csv.reader(f, delimiter="\t")
        header = next(tsvreader)
        for line in tsvreader:
            tsv.append([float(val) for val in line])

    json_dict = read_json(jsonfilename)

    return PETBIDSMatrix(np.array(tsv).T, json_dict, header)
