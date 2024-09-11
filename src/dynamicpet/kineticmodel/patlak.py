"""Patlak plotting."""

# References:
"""
https://osipi.github.io/DCE-DSC-MRI_TestResults/PatlakModel.html

https://journals.sagepub.com/doi/epdf/10.1038/jcbfm.1985.87

"""
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve  # type: ignore
from tqdm import trange

from ..temporalobject.temporalimage import TemporalImage
from ..temporalobject.temporalmatrix import TemporalMatrix
from ..typing_utils import NumpyNumberArray
from ..typing_utils import NumpyRealNumber
from .kineticmodel import KineticModel


class PATLAK(KineticModel):
    """ """

    @classmethod
    def get_param_names(cls) -> list[str]:
        """Get names of kinetic model parameters."""
        return ["slope"]

    def fit(
        self, mask: NumpyNumberArray[np.Any, np.dtype[np.number[np.Any]]] | None = None
    ) -> None:
        """ """
        tacs: TemporalMatrix = self.tacs.timeseries_in_mask(mask)
        reftac: TemporalMatrix = self.reftac.timeseries_in_mask(mask)
        num_elements = tacs.num_elements
        int_reftac: NumpyNumberArray = self.reftac.cumulative_integral(
            integration_type
        ).flatten()

        x = np.column_stack((np.ones_like(int_reftac), int_reftac / reftac.dataobj))
        denominator = reftac.dataobj
        w = np.ones_like(x)
        slope = np.zeros((num_elements, 1))
        for k in trange(num_elements):
            # get single TAC and its
            tac = tacs.dataobj[k, :]
            y = tac / denominator
            b: NumpyNumberArray
            b = solve(x.T @ w @ x, x.T @ w @ y, assume_a="sym")
            slope[k] = b[1]
        self.set_parameter("slope", slope, mask)
