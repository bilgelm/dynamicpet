"""Logan plot."""

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve  # type: ignore
from tqdm import trange

from ..temporalobject.temporalimage import TemporalImage
from ..temporalobject.temporalmatrix import TemporalMatrix
from ..temporalobject.temporalobject import INTEGRATION_TYPE_OPTS
from ..temporalobject.temporalobject import WEIGHT_OPTS
from ..typing_utils import NumpyNumberArray
from ..typing_utils import RealNumber
from .kineticmodel import KineticModel


class LoganPlot(KineticModel):
    """Logan Plot.

    Reference:
    Logan, J., Fowler, J. S., Volkow, N. D., Wang, G. J.,
    Ding, Y. S., & Alexoff, D. L. (1996).
    Distribution volume ratios without blood sampling
    from graphical analysis of PET data.
    Journal of Cerebral Blood Flow & Metabolism, 16(5), 834-840.
    """

    @classmethod
    def get_param_names(cls) -> list[str]:
        """Get names of kinetic model parameters."""
        return [
            "dvr",
            # "r1",
            # "k2",
            # "k2a",
            # "r1_lrsc",
            # "k2_lrsc",
            # "k2a_lrsc",
            "noise_var_eq_dvr",
            # "noise_var_eq_r1",
        ]

    def fit(  # noqa: max-complexity: 12
        self,
        mask: NumpyNumberArray | None = None,
        integration_type: INTEGRATION_TYPE_OPTS = "trapz",
        weight_by: WEIGHT_OPTS | NumpyNumberArray | None = "frame_duration",
        fwhm: RealNumber | list[RealNumber] | None = None,
    ) -> None:
        """Estimate model parameters.

        Args:
            integration_type: If 'rect', rectangular integration is used for TACs.
                              If 'trapz', trapezoidal integration is used based
                              on middle timepoint of each frame.
            weight_by: [optional] frame weights used in model fitting.
                       If weight_by == None, each frame is weighted equally.
                       If weight_by == 'frame_duration', each frame is weighted
                       proportionally to its duration (inverse variance weighting).
                       If weight_by is a 1-D array, then specified values are used.
            mask: [optional] A 1-D (for TemporalMatrix TACs) or
                  3-D (for TemporalImage TACs) binary mask that defines where
                  to fit the kinetic model. Elements outside the mask will
                  be set to to 0 in parametric estimate outputs.
            fwhm: scalar or length 3 sequence, FWHM in mm over which to smooth
        """
        # get reference TAC as a 1-D vector
        reftac: NumpyNumberArray = self.reftac.dataobj.flatten()[:, np.newaxis]
        # numerical integration of reference TAC
        int_reftac: NumpyNumberArray = self.reftac.cumulative_integral(
            integration_type
        ).flatten()

        tacs: TemporalMatrix = self.tacs.timeseries_in_mask(mask)
        num_elements = tacs.num_elements
        tacs_mat: NumpyNumberArray = tacs.dataobj
        int_tacs_mat: NumpyNumberArray = tacs.cumulative_integral(integration_type)

        weights = tacs.get_weights(weight_by)
        w = np.diag(weights)

        dvr = np.zeros((num_elements, 1))

        for k in trange(num_elements):
            # get TAC and its cumulative integral as 1-D vectors
            tac = tacs_mat[k, :][:, np.newaxis]

            # special case when tac is the same as reftac
            if np.allclose(tac, reftac):
                dvr[k] = 1

                continue

            int_tac = int_tacs_mat[k, :][:, np.newaxis]

            # ----- Get DVR -----
            # Set up the weighted linear regression model based on Eq. 7 in Logan et al.

            x = np.column_stack((np.ones(tac.shape), np.divide(int_reftac, tac)))
            y = np.divide(int_tac, tac)

            b: NumpyNumberArray
            try:
                b = solve(x.T @ w @ x, x.T @ w @ y, assume_a="sym")
            except LinAlgError:
                b = np.ones((2, 1))

            # distribution volume ratio
            dvr[k] = b[1]

        self.set_parameter("dvr", dvr, mask)

    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        """Get fitted TACs based on estimated model parameters."""
        # there is no parametric model for SUVR, so we just return the tacs
        return self.tacs
