"""Patlak plot."""

# References:
# https://osipi.github.io/DCE-DSC-MRI_TestResults/PatlakModel.html
# https://journals.sagepub.com/doi/epdf/10.1038/jcbfm.1985.87

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve  # type: ignore
from tqdm import trange

from ..temporalobject.temporalimage import TemporalImage
from ..temporalobject.temporalmatrix import TemporalMatrix
from ..temporalobject.temporalobject import INTEGRATION_TYPE_OPTS
from ..temporalobject.temporalobject import WEIGHT_OPTS
from ..typing_utils import NumpyNumberArray
from .kineticmodel import KineticModel


class PRTM(KineticModel):
    """Patlak reference tissue model.

    Also known as Patlak plot, Gjedde-Patlak plot, Patlak-Rutland plot,
    graphical Patlak, Patlak graphical method (or some permutation thereof these
    words) with reference tissue.
    "Non-invasive" can also be used instead of "with reference tissue" to convey
    the same meaning.

    Reference:
    Patlak CS, Blasberg RG. Graphical evaluation of blood-to-brain transfer
    constants from multiple-time uptake data. Generalizations. J Cereb Blood
    Flow Metab. 1985 Dec;5(4):584-90.
    """

    @classmethod
    def get_param_names(cls) -> list[str]:
        """Get names of kinetic model parameters."""
        return ["slope", "intercept"]

    def fit(
        self,
        mask: NumpyNumberArray | None = None,
        integration_type: INTEGRATION_TYPE_OPTS = "trapz",
        weight_by: WEIGHT_OPTS | NumpyNumberArray | None = "frame_duration",
        tstar: float = 0,
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
            tstar: time beyond which to assume linearity
        """
        # get reference TAC as a 1-D vector
        reftac: NumpyNumberArray = self.reftac.dataobj.flatten()[:, np.newaxis]
        # numerical integration of reference TAC
        int_reftac: NumpyNumberArray = self.reftac.cumulative_integral(
            integration_type
        ).flatten()[:, np.newaxis]

        tacs: TemporalMatrix = self.tacs.timeseries_in_mask(mask)
        num_elements = tacs.num_elements
        tacs_mat: NumpyNumberArray = tacs.dataobj

        # time indexing should be done after integrating
        t_idx = tacs.frame_start >= tstar
        reftac_tstar = reftac[t_idx, :]
        int_reftac_tstar = int_reftac[t_idx, :]
        tacs_mat_tstar = tacs_mat[:, t_idx]

        x = np.column_stack(
            (np.ones_like(int_reftac_tstar), int_reftac_tstar / reftac_tstar)
        )
        weights = tacs.get_weights(weight_by)
        w = np.diag(weights[t_idx])

        slope = np.ones((num_elements, 1))
        intercept = np.zeros((num_elements, 1))

        for k in trange(num_elements):
            # get TAC as 1-D vector
            tac_tstar = tacs_mat_tstar[k, :][:, np.newaxis]

            y = tac_tstar / reftac_tstar
            b: NumpyNumberArray
            try:
                b = solve(x.T @ w @ x, x.T @ w @ y, assume_a="sym")
                intercept[k], slope[k] = b
            except LinAlgError:
                pass

        self.set_parameter("slope", slope, mask)
        self.set_parameter("intercept", intercept, mask)

    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        """Get fitted TACs based on estimated model parameters."""
        raise NotImplementedError()
