"""Wrapper for kinfitr."""

from abc import ABC
from typing import List
from typing import Union

import numpy as np
from rpy2.robjects import default_converter  # type: ignore
from rpy2.robjects import numpy2ri
from rpy2.robjects import r
from rpy2.robjects.packages import importr  # type: ignore
from tqdm import trange  # type: ignore

from ..temporalobject.temporalimage import TemporalImage
from ..temporalobject.temporalmatrix import TemporalMatrix
from ..typing_utils import NumpyRealNumberArray
from .kineticmodel import KineticModel


np_cv_rules = default_converter + numpy2ri.converter


class KinfitrModel(KineticModel, ABC):
    """Generic wrapper for kinfitr reference tissue models."""

    @classmethod
    def get_r_name(cls) -> str:
        """Get the R function name."""
        return cls.__name__.lower()

    @classmethod
    def get_param_names(cls) -> List[str]:
        """Get names of kinetic model parameters."""
        return ["bp", "R1", "k2"]

    def fit(
        self,
        mask: NumpyRealNumberArray | None = None,
        **kwargs: Union[NumpyRealNumberArray, float]
    ) -> None:
        """Estimate model parameters.

        Args:
            mask: [optional] A 1-D (for TemporalMatrix TACs) or
                  3-D (for TemporalImage TACs) binary mask that defines where
                  to fit the kinetic model. Elements outside the mask will
                  be set to to NA in parametric estimate outputs.
            kwargs: optional arguments for the kinfitr function
        """
        tacs: TemporalMatrix = self.tacs.timeseries_in_mask(mask)
        num_elements = tacs.num_elements
        t_tac = self.reftac.frame_mid.flatten()
        reftac = self.reftac.dataobj.flatten()
        roitacs = tacs.dataobj.reshape(num_elements, tacs.num_frames)

        importr("kinfitr")
        param_estimates = {}

        with np_cv_rules.context():
            for i in trange(num_elements):
                kinfitr_fun = r[self.__class__.get_r_name()]
                res = kinfitr_fun(t_tac, reftac, roitacs[i, :].flatten(), **kwargs)
                for param_name in res["par"].dtype.names:
                    if param_name not in param_estimates:
                        param_estimates[param_name] = np.zeros((num_elements, 1))
                    param_estimates[param_name][i] = res["par"][param_name]

        for param_name, param_estimate in param_estimates.items():
            self.set_parameter(param_name, param_estimate, mask)

    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        """Get fitted TACs based on estimated model parameters."""
        raise NotImplementedError()


class FRTM(KinfitrModel):
    """kinfitr frtm wrapper."""

    @classmethod
    def get_param_names(cls) -> List[str]:
        """Get names of kinetic model parameters."""
        return ["bp", "R1", "k2", "k3", "k4"]


class SRTM(KinfitrModel):
    """kinfitr srtm wrapper."""

    pass


class SRTM2(KinfitrModel):
    """kinfitr srtm2 wrapper."""

    @classmethod
    def get_param_names(cls) -> List[str]:
        """Get names of kinetic model parameters."""
        return ["bp", "R1", "k2", "k2a"]


class MRTM1(KinfitrModel):
    """kinfitr mrtm1 wrapper."""

    @classmethod
    def get_param_names(cls) -> List[str]:
        """Get names of kinetic model parameters."""
        return ["bp", "k2prime", "R1", "k2"]


class MRTM2(KinfitrModel):
    """kinfitr mrtm2 wrapper."""

    pass


class RefLogan(KinfitrModel):
    """kinfitr refLogan wrapper."""

    def __init__(self) -> None:
        """Placeholder for raising error."""
        raise NotImplementedError("Doesn't work due to rpy2 error")

    @classmethod
    def get_r_name(cls) -> str:
        """Get the R function name."""
        return "refLogan"

    @classmethod
    def get_param_names(cls) -> List[str]:
        """Get names of kinetic model parameters."""
        return ["bp"]


class RefMLLogan(KinfitrModel):
    """kinfitr refmlLogan wrapper."""

    def __init__(self) -> None:
        """Placeholder for raising error."""
        raise NotImplementedError()


class RefPatlak(KinfitrModel):
    """kinfitr refPatlak wrapper."""

    def __init__(self) -> None:
        """Placeholder for raising error."""
        raise NotImplementedError()
