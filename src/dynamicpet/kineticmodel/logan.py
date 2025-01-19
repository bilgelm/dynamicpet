"""Logan plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve  # type: ignore[import-untyped]
from tqdm import trange

from dynamicpet.kineticmodel.kineticmodel import KineticModel

if TYPE_CHECKING:
    from dynamicpet.temporalobject.temporalimage import TemporalImage
    from dynamicpet.temporalobject.temporalmatrix import TemporalMatrix
    from dynamicpet.temporalobject.temporalobject import (
        INTEGRATION_TYPE_OPTS,
        WEIGHT_OPTS,
    )
    from dynamicpet.typing_utils import NumpyNumberArray


class LRTM(KineticModel):
    """Logan reference tissue model.

    Also known as Logan plot, graphical Logan, Logan graphical analysis (or
    some permutation thereof these words) with reference tissue.
    "Non-invasive" can also be used instead of "with reference tissue" to convey
    the same meaning.

    Reference:
    Logan J, Fowler JS, Volkow ND, Wang GJ, Ding YS, Alexoff DL. Distribution
    volume ratios without blood sampling from graphical analysis of PET data.
    J Cereb Blood Flow Metab. 1996 Sep;16(5):834-40.
    """

    @classmethod
    def get_param_names(cls) -> list[str]:
        """Get names of kinetic model parameters."""
        return ["DVR"]

    def fit(  # noqa: max-complexity: 12
        self,
        mask: NumpyNumberArray | None = None,
        integration_type: INTEGRATION_TYPE_OPTS = "trapz",
        weight_by: WEIGHT_OPTS | NumpyNumberArray | None = "frame_duration",
        tstar: float = 0,
        k2prime: float | None = None,
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
            k2prime: (avg.) effective tissue-to-plasma efflux constant in the
                     reference region, in unit of 1/min

        """
        # get reference TAC as a 1-D vector
        reftac: NumpyNumberArray = self.reftac.dataobj.flatten()[:, np.newaxis]
        # numerical integration of reference TAC
        int_reftac: NumpyNumberArray = self.reftac.cumulative_integral(
            integration_type,
        ).flatten()[:, np.newaxis]

        tacs: TemporalMatrix = self.tacs.timeseries_in_mask(mask)
        num_elements = tacs.num_elements
        tacs_mat: NumpyNumberArray = tacs.dataobj
        int_tacs_mat: NumpyNumberArray = tacs.cumulative_integral(integration_type)

        # time indexing should be done after integrating
        t_idx = tacs.frame_start >= tstar
        reftac_tstar = reftac[t_idx, :]
        int_reftac_tstar = int_reftac[t_idx, :]
        tacs_mat_tstar = tacs_mat[:, t_idx]
        int_tacs_mat_tstar = int_tacs_mat[:, t_idx]

        weights = tacs.get_weights(weight_by)
        w_star = np.diag(weights[t_idx])

        dvr = np.zeros((num_elements, 1))

        if not k2prime:
            # TODO @bilgelm: sanity checks  # noqa: TD003, FIX002
            # Check Eq. 7 assumption (i.e., that tac / reftac is reasonably
            # constant) by calculating R2 etc. for each tac.
            # Display warning if assumption is off.
            pass

        for k in trange(num_elements):
            # get TAC and its cumulative integral as 1-D vectors
            tac_tstar = tacs_mat_tstar[k, :][:, np.newaxis]

            # special case when tac is the same as reftac
            if np.allclose(tac_tstar, reftac_tstar):
                dvr[k] = 1

                continue

            int_tac_tstar = int_tacs_mat_tstar[k, :][:, np.newaxis]

            # ----- Get DVR -----
            # Set up the weighted linear regression model based on Logan et al.:
            # - use Eq. 6 if k2prime is provided
            # - use Eq. 7 if k2prime is not provided

            x = np.column_stack(
                (
                    np.ones_like(tac_tstar),
                    (int_reftac_tstar + (reftac_tstar / k2prime if k2prime else 0))
                    / tac_tstar,
                ),
            )
            y = int_tac_tstar / tac_tstar

            b: NumpyNumberArray
            try:
                b = solve(x.T @ w_star @ x, x.T @ w_star @ y, assume_a="sym")
            except LinAlgError:
                b = np.ones((2, 1))

            # distribution volume ratio
            dvr[k] = b[1]

        self.set_parameter("DVR", dvr, mask)
        # should tstar (and k2prime?) also be stored?

    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        """Get fitted TACs based on estimated model parameters."""
        raise NotImplementedError
