"""Kinetic modeling for dynamic PET (for reference tissue models)."""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import numpy as np
from nibabel.spatialimages import SpatialImage

from ..petbids.petbidsmatrix import PETBIDSMatrix
from ..temporalobject.temporalimage import TemporalImage
from ..temporalobject.temporalmatrix import TemporalMatrix
from ..temporalobject.temporalobject import TemporalObject
from ..typing_utils import NumpyRealNumberArray


# rename to ReferenceTissueKineticModel ??
class KineticModel(ABC):
    """KineticModel abstract based class.

    Attributes:
        reftac: reference time activity curve (TAC)
        tacs: TACs in regions/voxels of interest
        parameters: kinetic model parameters
    """

    reftac: TemporalMatrix | PETBIDSMatrix
    tacs: TemporalObject[Any]
    parameters: Dict[str, NumpyRealNumberArray]

    def __init__(
        self, reftac: TemporalMatrix | PETBIDSMatrix, tacs: TemporalObject[Any]
    ) -> None:
        """Initialize a kinetic model.

        Args:
            reftac: reference time activity curve (TAC), must have single element
            tacs: TACs in regions/voxels of interest, with num_frames equal to
                  that of reftac

        Raises:
            ValueError: incompatible temporal dimensions or non-finite TACs
        """
        # basic input checks
        if reftac.num_elements != 1:
            raise ValueError("Reference TAC is not 1-D")
        if not np.all(np.isfinite(reftac.dataobj)):
            raise ValueError("Reference TAC has non-finite value(s)")
        if not np.all(np.isfinite(tacs.dataobj)):
            raise ValueError("TAC(s) has/have non-finite value(s)")

        if not tacs.num_frames == reftac.num_frames:
            raise ValueError("reftac and tacs must have same length")
        if not np.all(tacs.frame_start == reftac.frame_start):
            raise ValueError("reftac and tacs should have same frame starts")
        if not np.all(tacs.frame_duration == reftac.frame_duration):
            raise ValueError("reftac and tacs should have same frame ends")

        self.reftac: TemporalMatrix | PETBIDSMatrix = reftac
        self.tacs: TemporalObject[Any] = tacs
        self.parameters: Dict[str, NumpyRealNumberArray] = {}

    @abstractmethod
    def fit(self) -> None:
        """Estimate model parameters."""
        # implementation should update self.results
        pass

    def get_parameter(self, param_name: str) -> SpatialImage | NumpyRealNumberArray:
        """Get a fitted parameter.

        If the input (tacs) is an image, parameter will be returned as an image.
        Otherwise, it will be a matrix.

        Args:
            param_name: name of parameter

        Returns:
            parameter matrix or image

        Raises:
            AttributeError: no estimate is available (kinetic model has not
                            been fitted) or this model doesn't have such a
                            parameter
        """
        if param_name in self.parameters:
            # if tacs is an image and number of parameter estimates is equal
            # to the number of image voxels, then return parameter as an image;
            # otherwise, return as a matrix
            if (
                isinstance(self.tacs, TemporalImage)
                and self.tacs.num_voxels == self.parameters[param_name].size
            ):
                image_maker = self.tacs.img.__class__
                param_img: SpatialImage = image_maker(
                    self.parameters[param_name],
                    self.tacs.img.affine,
                    self.tacs.img.header,
                )
                return param_img
            else:
                param_vector: NumpyRealNumberArray = self.parameters[param_name]
                return param_vector
        elif param_name == "bp" and "dvr" in self.parameters:
            self.parameters[param_name] = self.parameters["dvr"] - 1
            return self.get_parameter(param_name)
        elif param_name == "dvr" and "bp" in self.parameters:
            self.parameters[param_name] = self.parameters["bp"] + 1
            return self.get_parameter(param_name)
        else:
            raise AttributeError(
                "No estimate available for parameter " f"{param_name}."
            )
