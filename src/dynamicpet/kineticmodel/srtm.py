"""Kinetic modeling for dynamic PET using SRTM by Zhou et al. (2003)."""

from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg import solve

from ..temporalobject.temporalmatrix import TemporalMatrix
from ..temporalobject.temporalobject import WEIGHT_OPTS
from ..temporalobject.temporalobject import TemporalObject
from ..typing_utils import NumpyRealNumberArray
from .kineticmodel import KineticModel


class SRTMZhou2003(KineticModel):
    """Simplified reference tissue model (SRTM) with linear spatial constraint.

    Attributes:
        weights: vector of weights with num_frames elements
    """

    weights: NumpyRealNumberArray

    def __init__(
        self,
        reftac: TemporalMatrix,
        tacs: TemporalObject[Any],
        weight_by: WEIGHT_OPTS | NumpyRealNumberArray | None = "frame_duration",
    ) -> None:
        """Initialize a SRTM-LRSC kinetic model.

        Args:
            reftac: reference time activity curve (TAC), must have single element
            tacs: TACs in regions/voxels of interest, with num_frames equal to
                  that of reftac
            weight_by: If weight_by == None, each frame is weighted equally.
                       If weight_by == 'frame_duration', each frame is weighted
                       proportionally to its duration (inverse variance weighting).
                       If weight_by is a 1-D array, then specified values are used.
        """
        super().__init__(reftac, tacs)

        self.weights: NumpyRealNumberArray = tacs.get_weights(weight_by)

    def fit(self) -> None:
        """Estimate model parameters."""
        # get reference TAC as a 1-D vector
        reftac = self.reftac.dataobj.flatten()[:, np.newaxis]
        # numerical integration of reference TAC
        int_reftac = self.reftac.cumulative_integral().flatten()

        w = np.diag(self.weights)

        n = self.tacs.num_frames
        m = 3

        num_elements = self.tacs.num_elements
        tacs_mat = np.reshape(self.tacs.dataobj, (num_elements, n))
        int_tacs = self.tacs.cumulative_integral()
        int_tacs_mat = np.reshape(int_tacs, (num_elements, n))

        dvr = np.zeros((num_elements, 1))
        noise_var_eq_dvr = np.zeros((num_elements, 1))
        r1 = np.zeros((num_elements, 1))
        k2 = np.zeros((num_elements, 1))
        k2a = np.zeros((num_elements, 1))
        noise_var_eq_r1 = np.zeros((num_elements, 1))

        for k in range(num_elements):
            # get TAC and its cumulative integral as 1-D vectors
            tac = tacs_mat[k, :][:, np.newaxis]
            int_tac = int_tacs_mat[k, :][:, np.newaxis]

            # ----- Get DVR -----
            # Set up the weighted linear regression model
            # based on Eq. 9 in Zhou et al.
            # Per the recommendation in 1st paragraph on p. 979 of Zhou et al.,
            # smoothed TAC is used in the design matrix, if provided.
            x = np.column_stack((int_reftac, reftac, -tac))

            b: NumpyRealNumberArray
            try:
                b = solve(x.T @ w @ x, x.T @ w @ int_tac)
            except LinAlgError:
                b = np.zeros((3, 1))

            residual = int_tac - x @ b

            # unbiased estimator of noise variance
            noise_var_eq_dvr[k] = residual.T @ w @ residual / (n - m)

            # distribution volume ratio
            dvr[k] = b[0]

            # ----- Get R1 -----
            # Set up the weighted linear regression model
            # based on Eq. 8 in Zhou et al.
            x = np.column_stack((reftac, int_reftac, -int_tac))
            try:
                b = solve(x.T @ w @ x, x.T @ w @ tac)
            except LinAlgError:
                b = np.zeros((3, 1))

            residual = tac - x @ b
            noise_var_eq_r1[k] = residual.T @ w @ residual / (n - m)

            r1[k], k2[k], k2a[k] = b

        self.parameters["dvr"] = np.reshape(dvr, self.tacs.shape[:-1])
        self.parameters["r1"] = np.reshape(r1, self.tacs.shape[:-1])
        self.parameters["k2"] = np.reshape(k2, self.tacs.shape[:-1])
        self.parameters["k2a"] = np.reshape(k2a, self.tacs.shape[:-1])
        self.parameters["noise_var_eq_dvr"] = np.reshape(
            noise_var_eq_dvr, self.tacs.shape[:-1]
        )
        self.parameters["noise_var_eq_r1"] = np.reshape(
            noise_var_eq_r1, self.tacs.shape[:-1]
        )
