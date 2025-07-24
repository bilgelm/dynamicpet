"""Kinetic modeling for dynamic PET (for reference tissue models)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from dynamicpet.temporalobject.temporalimage import TemporalImage, image_maker

if TYPE_CHECKING:
    from nibabel.spatialimages import SpatialImage

    from dynamicpet.temporalobject.temporalmatrix import TemporalMatrix
    from dynamicpet.typing_utils import NumpyNumberArray


# rename to ReferenceTissueKineticModel ??
class KineticModel(ABC):
    """KineticModel abstract base class.

    Args:
        reftac: reference time activity curve (TAC), must have single element
        tacs: TACs in regions/voxels of interest, with num_frames equal to
              that of reftac

    Attributes:
        reftac: reference time activity curve (TAC), with times specified in minutes
        tacs: TACs in regions/voxels of interest, with times specified in minutes
        parameters: kinetic model parameters, of same spatial dimension as input TACs

    """

    reftac: TemporalMatrix
    tacs: TemporalMatrix | TemporalImage
    parameters: dict[str, NumpyNumberArray]

    @classmethod
    @abstractmethod
    def get_param_names(cls) -> list[str]:
        """Get names of kinetic model parameters."""
        # parameter names should contain alphanumeric characters only
        raise NotImplementedError

    def __init__(
        self,
        reftac: TemporalMatrix,
        tacs: TemporalMatrix | TemporalImage,
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
            msg = "Reference TAC is not 1-D"
            raise ValueError(msg)
        if not np.all(np.isfinite(reftac.dataobj)):
            msg = "Reference TAC has non-finite value(s)"
            raise ValueError(msg)
        if not np.all(np.isfinite(tacs.dataobj)):
            msg = "TAC(s) has/have non-finite value(s)"
            raise ValueError(msg)

        if tacs.num_frames != reftac.num_frames:
            msg = "reftac and tacs must have same length"
            raise ValueError(msg)
        if not np.all(tacs.frame_start == reftac.frame_start):
            msg = "reftac and tacs should have same frame starts"
            raise ValueError(msg)
        if not np.all(tacs.frame_duration == reftac.frame_duration):
            msg = "reftac and tacs should have same frame ends"
            raise ValueError(msg)

        self.reftac: TemporalMatrix = reftac
        self.tacs: TemporalMatrix | TemporalImage = tacs
        self.parameters: dict[str, NumpyNumberArray] = {}

    @abstractmethod
    def fit(self, mask: NumpyNumberArray | None = None) -> None:
        """Estimate model parameters."""
        # implementation should update self.parameters
        raise NotImplementedError

    def get_parameter(self, param_name: str) -> SpatialImage | NumpyNumberArray:
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
            if isinstance(self.tacs, TemporalImage):
                param_img: SpatialImage = image_maker(
                    self.parameters[param_name],
                    self.tacs.img,
                )
                return param_img
            param_vector: NumpyNumberArray = self.parameters[param_name]
            return param_vector
        if param_name == "BPND" and "DVR" in self.parameters:
            self.parameters[param_name] = self.parameters["DVR"] - 1
            return self.get_parameter(param_name)
        if param_name == "DVR" and "BPND" in self.parameters:
            self.parameters[param_name] = self.parameters["BPND"] + 1
            return self.get_parameter(param_name)

        msg = f"No estimate available for parameter {param_name}."
        raise AttributeError(msg)

    def set_parameter(
        self,
        param_name: str,
        param: NumpyNumberArray,
        mask: NumpyNumberArray | None = None,
    ) -> None:
        """Set kinetic model parameter.

        Args:
            param_name: name of parameter to set
            param: parameter estimate
            mask: [optional] A 1-D (for TemporalMatrix TACs) or
                  3-D (for TemporalImage TACs) binary mask that defines where
                  the kinetic model was fitted. Elements outside the mask will
                  be set to to 0 in parametric outputs.

        """
        if mask is None:
            if hasattr(param, "size") and param.size == self.tacs.num_elements:
                self.parameters[param_name] = np.reshape(
                    param,
                    self.tacs.dataobj.shape[:-1],
                )
            else:
                self.parameters[param_name] = param
        else:
            tmp = np.zeros_like(self.tacs.dataobj[..., 0])
            tmp.flat[mask.astype("bool").flatten()] = param
            self.parameters[param_name] = tmp

    @abstractmethod
    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        """Get fitted TACs based on estimated model parameters."""
