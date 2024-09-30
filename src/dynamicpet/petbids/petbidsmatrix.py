"""PETBIDSMatrix class."""

import csv
import os.path as op
from copy import deepcopy
from os import PathLike
from typing import Literal

import numpy as np

from ..temporalobject.temporalmatrix import TemporalMatrix
from ..typing_utils import NumpyNumberArray
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
        dataobj: NumpyNumberArray,
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
            start_time: time (min) at which to begin relative to TimeZero, incl.
            end_time: time (min) at which to stop relative to TimeZero, incl.

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
        """
        newdecaycorrecttime, original_anchor = self._decay_correct_offset(other)
        other = other.decay_correct(decaycorrecttime=newdecaycorrecttime)

        concat_mat = super().concatenate(other)
        json_dict = update_frametiming_from(self.json_dict, concat_mat)

        concat_res = PETBIDSMatrix(concat_mat.dataobj, json_dict)
        concat_res.set_timezero(anchor=original_anchor)

        return concat_res

    def decay_correct(self, decaycorrecttime: float = 0) -> "PETBIDSMatrix":
        """Return decay corrected PETBIDSMatrix.

        Args:
            decaycorrecttime: time to decay correct to, relative to time zero

        Returns:
            decay corrected TACs
        """
        tacs = self.get_decay_corrected_tacs(decaycorrecttime)
        corrected_tacs = np.reshape(tacs, self.shape)

        json_dict = deepcopy(self.json_dict)
        json_dict["ImageDecayCorrected"] = True
        json_dict["ImageDecayCorrectionTime"] = (
            decaycorrecttime + json_dict["ScanStart"] + json_dict["InjectionStart"]
        )

        return PETBIDSMatrix(corrected_tacs, json_dict)

    def decay_uncorrect(self) -> "PETBIDSMatrix":
        """Return decay uncorrected PETBIDSMatrix."""
        tacs = self.get_decay_uncorrected_tacs()
        uncorrected_tacs = np.reshape(tacs, self.shape)

        json_dict = deepcopy(self.json_dict)
        json_dict["ImageDecayCorrected"] = False
        # PET-BIDS still requires "ImageDecayCorrectionTime" tag, so we don't
        # do anything about it

        return PETBIDSMatrix(uncorrected_tacs, json_dict)

    def to_filename(
        self,
        filename: str | PathLike[str],
        save_json: bool = False,
        anchor: Literal["InjectionStart", "ScanStart"] = "InjectionStart",
    ) -> None:
        """Save to file.

        Args:
            filename: file name for the tabular TAC tsv output
            save_json: whether the PET-BIDS json side car should be saved
            anchor: time anchor. The corresponding tag in the PET-BIDS json will
                    be set to zero (with appropriate offsets applied to other
                    tags).

        Raises:
            ValueError: file is not a tsv file
        """
        with open(filename, "w") as f:
            tsvwriter = csv.writer(f, delimiter="\t")
            tsvwriter.writerow(self.elem_names)
            for row in self.dataobj.T:
                tsvwriter.writerow(row)

        if save_json:
            self.set_timezero(anchor)

            fbase, fext = op.splitext(filename)
            if fext != ".tsv":
                raise ValueError("output file must be a tsv file")
            jsonfilename = fbase + ".json"
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
    json_dict = read_json(jsonfilename)

    with open(filename) as f:
        tsvreader = csv.reader(f, delimiter="\t")
        header = next(tsvreader)

    tsv = np.genfromtxt(filename, delimiter="\t", skip_header=1)

    return PETBIDSMatrix(tsv.T, json_dict, header)
