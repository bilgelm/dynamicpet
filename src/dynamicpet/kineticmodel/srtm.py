"""Simplified reference tissue model (SRTM)."""

import numpy as np
from nibabel.processing import smooth_image
from nibabel.spatialimages import SpatialImage
from numpy.linalg import LinAlgError
from scipy.linalg import solve  # type: ignore
from scipy.optimize import curve_fit  # type: ignore
from scipy.signal import convolve  # type: ignore
from tqdm import trange

from ..temporalobject.temporalimage import TemporalImage
from ..temporalobject.temporalimage import image_maker
from ..temporalobject.temporalmatrix import TemporalMatrix
from ..temporalobject.temporalobject import INTEGRATION_TYPE_OPTS
from ..temporalobject.temporalobject import WEIGHT_OPTS
from ..typing_utils import NumpyNumberArray
from ..typing_utils import RealNumber
from .kineticmodel import KineticModel


class SRTMLammertsma1996(KineticModel):
    """Simplified reference tissue model (SRTM).

    Reference:
    Lammertsma AA, Hume SP. Simplified reference tissue model for PET receptor
    studies. NeuroImage. 1996 Dec;4(3 Pt 1):153-8.
    """

    @classmethod
    def get_param_names(cls) -> list[str]:
        """Get names of kinetic model parameters."""
        return ["BP_ND", "R1", "k2"]

    def fit(
        self,
        mask: NumpyNumberArray | None = None,
        weight_by: WEIGHT_OPTS | NumpyNumberArray | None = None,
    ) -> None:
        """Estimate model parameters.

        Args:
            weight_by: [optional] frame weights used in model fitting.
                       If weight_by == None, each frame is weighted equally.
                       If weight_by == 'frame_duration', each frame is weighted
                       proportionally to its duration (inverse variance weighting).
                       If weight_by is a 1-D array, then specified values are used.
            mask: [optional] A 1-D (for TemporalMatrix TACs) or
                  3-D (for TemporalImage TACs) binary mask that defines where
                  to fit the kinetic model. Elements outside the mask will
                  be set to to 0 in parametric estimate outputs.
        """
        tacs: TemporalMatrix = self.tacs.timeseries_in_mask(mask)
        num_elements = tacs.num_elements
        roitacs = tacs.dataobj.reshape(num_elements, tacs.num_frames)

        weights = tacs.get_weights(weight_by)

        bp_nd = np.zeros((num_elements, 1))
        r1 = np.zeros((num_elements, 1))
        k2 = np.zeros((num_elements, 1))
        for k in trange(num_elements):
            init_guess = (1.5, 1.0, 0.1)
            popt, _ = curve_fit(
                srtm_model,
                self.reftac,
                roitacs[k, :].flatten(),
                init_guess,
                sigma=weights,
                bounds=([0, 0, 0], [15, 10, 1]),
            )
            bp_nd[k], r1[k], k2[k] = popt

        self.set_parameter("BP_ND", bp_nd, mask)
        self.set_parameter("R1", r1, mask)
        self.set_parameter("k2", k2, mask)

    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        """Get fitted TACs based on estimated model parameters."""
        num_elements = self.tacs.num_elements
        fitted_tacs_dataobj = np.empty_like(self.tacs.dataobj)

        for i in trange(num_elements):
            idx = np.unravel_index(i, self.tacs.shape[:-1])
            bp_nd = self.parameters["BP_ND"][*idx]
            r1 = self.parameters["R1"][*idx]
            k2 = self.parameters["k2"][*idx]
            if bp_nd or r1 or k2:
                fitted_tacs_dataobj[*idx, :] = srtm_model(self.reftac, bp_nd, r1, k2)

        if isinstance(self.tacs, TemporalImage):
            img = image_maker(fitted_tacs_dataobj, self.tacs.img)
            ti = TemporalImage(img, self.tacs.frame_start, self.tacs.frame_duration)
            return ti
        else:
            tm = TemporalMatrix(
                fitted_tacs_dataobj, self.tacs.frame_start, self.tacs.frame_duration
            )
            return tm


def srtm_model(
    reftac: TemporalMatrix, bp_nd: float, r1: float, k2: float
) -> NumpyNumberArray:
    """SRTM model to generate a target TAC.

    Args:
        reftac: reference TAC
        bp_nd: binding potential
        r1: relative radiotracer delivery parameter, R1
        k2: k2

    Returns:
        target TAC
    """
    # because reftac frames are not necessarily evenly spaced,
    # we cannot use convolve. To resolve this issue, we first upsample
    # to a higher frequency uniform timing grid, convolve, and then
    # downsample to original timing grid

    t = reftac.frame_mid.astype("float")
    # find smallest time interval in data
    # step = np.min(reftac.frame_duration)
    # t_upsampled = np.arange(t[0], t[-1], step)
    # if t_upsampled[-1] < t[-1]:
    #     t_upsampled = np.append(t_upsampled, t_upsampled[-1] + step)
    t_upsampled, step = np.linspace(np.min(t), np.max(t), 1024, retstep=True)

    reftac_upsampled = np.interp(
        t_upsampled, t, reftac.dataobj.astype("float").flatten()
    )

    k2a = k2 / (1 + bp_nd)
    conv_res_upsampled = (
        convolve(reftac_upsampled, np.exp(-k2a * t_upsampled), mode="full")[
            : len(t_upsampled)
        ]
        * step
    )
    tac_upsampled = r1 * reftac_upsampled + (k2 - r1 * k2a) * conv_res_upsampled
    tac = np.interp(t, t_upsampled, tac_upsampled)

    return tac


class SRTMZhou2003(KineticModel):
    """Simplified reference tissue model (SRTM) with linear spatial constraint.

    Reference:
    Zhou Y, Endres CJ, Brašić JR, Huang S-C, Wong DF.
    Linear regression with spatial constraint to generate parametric images of
    ligand-receptor dynamic PET studies with a simplified reference tissue model.
    Neuroimage. 2003;18:975-989.
    """

    @classmethod
    def get_param_names(cls) -> list[str]:
        """Get names of kinetic model parameters."""
        return [
            "DVR",
            "R1",
            "k2",
            "k2a",
            "R1_lrsc",
            "k2_lrsc",
            "k2a_lrsc",
            "noise_var_eq_dvr",
            "noise_var_eq_r1",
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
        n = tacs.num_frames
        m = 3
        num_elements = tacs.num_elements
        tacs_mat: NumpyNumberArray = tacs.dataobj
        int_tacs_mat: NumpyNumberArray = tacs.cumulative_integral(integration_type)

        weights = tacs.get_weights(weight_by)
        w = np.diag(weights)

        # Per the recommendation in 1st paragraph on p. 979 of Zhou et al.,
        # smoothed TAC is used in the design matrix if an image + FWHM is provided.
        do_smooth = isinstance(self.tacs, TemporalImage) and fwhm is not None
        if do_smooth:
            smooth_img: SpatialImage = smooth_image(self.tacs.img, fwhm)  # type: ignore
            if mask is None:
                tacs_mat = np.reshape(smooth_img.get_fdata(), (num_elements, n))
            else:
                tacs_mat = smooth_img.get_fdata()[mask.astype("bool"), :]

        dvr = np.zeros((num_elements, 1))
        noise_var_eq_dvr = np.zeros((num_elements, 1))
        r1 = np.zeros((num_elements, 1))
        k2 = np.zeros((num_elements, 1))
        k2a = np.zeros((num_elements, 1))
        noise_var_eq_r1 = np.zeros((num_elements, 1))

        for k in trange(num_elements):
            # get TAC and its cumulative integral as 1-D vectors
            tac = tacs_mat[k, :][:, np.newaxis]

            # special case when tac is the same as reftac
            if np.allclose(tac, reftac):
                dvr[k] = 1
                r1[k] = 1
                k2[k] = np.nan
                k2a[k] = np.nan
                # noise_vars are 0 in this case as there is no error
                # since default is already 0, no need to specify
                continue

            int_tac = int_tacs_mat[k, :][:, np.newaxis]

            # ----- Get DVR -----
            # Set up the weighted linear regression model based on Eq. 9 in Zhou et al.
            # Per the recommendation in 1st paragraph on p. 979 of Zhou et al.,
            # smoothed TAC is used in the design matrix if an image +
            # a smoothing FWHM is provided.
            x = np.column_stack((int_reftac, reftac, -tac))

            b: NumpyNumberArray
            try:
                b = solve(x.T @ w @ x, x.T @ w @ int_tac, assume_a="sym")
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
                b = solve(x.T @ w @ x, x.T @ w @ tac, assume_a="sym")
            except LinAlgError:
                b = np.zeros((3, 1))

            residual = tac - x @ b
            noise_var_eq_r1[k] = residual.T @ w @ residual / (n - m)

            r1[k], k2[k], k2a[k] = b

        self.set_parameter("DVR", dvr, mask)
        self.set_parameter("R1", r1, mask)
        self.set_parameter("k2", k2, mask)
        self.set_parameter("k2a", k2a, mask)
        self.set_parameter("noise_var_eq_dvr", noise_var_eq_dvr, mask)
        self.set_parameter("noise_var_eq_r1", noise_var_eq_r1, mask)

        if do_smooth:
            smooth_r1_mat, smooth_k2_mat, smooth_k2a_mat, h = self.prep_refine_r1(
                mask, fwhm
            )

            r1_lrsc = np.zeros((num_elements, 1))
            k2_lrsc = np.zeros((num_elements, 1))
            k2a_lrsc = np.zeros((num_elements, 1))

            for k in trange(num_elements):
                # get TAC and its cumulative integral as 1-D vectors
                tac = tacs_mat[k, :][:, np.newaxis]
                int_tac = int_tacs_mat[k, :][:, np.newaxis]

                # ----- Get R1 incorporating spatial constraint -----
                # Set up the ridge regression model
                # based on Eq. 11 in Zhou et al.
                x = np.column_stack((reftac, int_reftac, -int_tac))
                h_d = np.diag(h[k, :])
                b_sc = np.vstack(
                    (smooth_r1_mat[k], smooth_k2_mat[k], smooth_k2a_mat[k])
                )
                try:
                    b = solve(
                        x.T @ w @ x + h_d, x.T @ w @ tac + h_d @ b_sc, assume_a="sym"
                    )
                except LinAlgError:
                    b = np.zeros((3, 1))

                r1_lrsc[k], k2_lrsc[k], k2a_lrsc[k] = b

            self.set_parameter("R1_lrsc", r1_lrsc, mask)
            self.set_parameter("k2_lrsc", k2_lrsc, mask)
            self.set_parameter("k2a_lrsc", k2a_lrsc, mask)

    def prep_refine_r1(
        self,
        mask: NumpyNumberArray | None = None,
        fwhm: RealNumber | list[RealNumber] | None = None,
    ) -> tuple[NumpyNumberArray, ...]:
        """Refine R1.

        Args:
            mask: [optional] A 1-D (for TemporalMatrix TACs) or
                  3-D (for TemporalImage TACs) binary mask that defines where
                  to fit the kinetic model. Elements outside the mask will
                  be set to to 0 in parametric estimate outputs.
            fwhm: scalar or length 3 sequence, FWHM in mm over which to smooth

        Returns:
            smooth_r1_mat: flattened matrix (according to mask) of smoothed r1
            smooth_k2_mat: flattened matrix (according to mask) of smoothed k2
            smooth_k2a_mat: flattened matrix (according to mask) of smoothed k2a
            h: matrix as described in Zhou et al.
        """
        m = 3

        smooth_r1_img: SpatialImage = smooth_image(
            self.get_parameter("R1"), fwhm  # type: ignore
        )
        smooth_k2_img: SpatialImage = smooth_image(
            self.get_parameter("k2"), fwhm  # type: ignore
        )
        smooth_k2a_img: SpatialImage = smooth_image(
            self.get_parameter("k2a"), fwhm  # type: ignore
        )

        noise_var_eq_r1_img: SpatialImage = self.get_parameter("noise_var_eq_r1")  # type: ignore
        r1_img: SpatialImage = self.get_parameter("R1")  # type: ignore
        k2_img: SpatialImage = self.get_parameter("k2")  # type: ignore
        k2a_img: SpatialImage = self.get_parameter("k2a")  # type: ignore
        noise_var_eq_r1_data = noise_var_eq_r1_img.get_fdata()
        # we add a small number to the denominator to prevent division by zero
        eps = np.finfo(float).eps
        h0 = (
            m
            * noise_var_eq_r1_data
            / (np.square(r1_img.get_fdata() - smooth_r1_img.get_fdata()) + eps)
        )
        h1 = (
            m
            * noise_var_eq_r1_data
            / (np.square(k2_img.get_fdata() - smooth_k2_img.get_fdata()) + eps)
        )
        h2 = (
            m
            * noise_var_eq_r1_data
            / (np.square(k2a_img.get_fdata() - smooth_k2a_img.get_fdata()) + eps)
        )

        h0_img = image_maker(h0, smooth_r1_img)
        h1_img = image_maker(h1, smooth_k2_img)
        h2_img = image_maker(h2, smooth_k2a_img)

        smooth_r1_mat: NumpyNumberArray
        smooth_k2_mat: NumpyNumberArray
        smooth_k2a_mat: NumpyNumberArray
        if mask is None:
            smooth_r1_mat = smooth_r1_img.get_fdata().flatten()
            smooth_k2_mat = smooth_k2_img.get_fdata().flatten()
            smooth_k2a_mat = smooth_k2a_img.get_fdata().flatten()

            num_elements = smooth_r1_mat.size
            h = np.zeros((num_elements, m))
            h[:, 0] = smooth_image(h0_img, fwhm).get_fdata().flatten()  # type: ignore
            h[:, 1] = smooth_image(h1_img, fwhm).get_fdata().flatten()  # type: ignore
            h[:, 2] = smooth_image(h2_img, fwhm).get_fdata().flatten()  # type: ignore
        else:
            smooth_r1_mat = smooth_r1_img.get_fdata()[mask.astype("bool")]
            smooth_k2_mat = smooth_k2_img.get_fdata()[mask.astype("bool")]
            smooth_k2a_mat = smooth_k2a_img.get_fdata()[mask.astype("bool")]

            num_elements = mask.astype("bool").sum()
            h = np.zeros((num_elements, m))
            h[:, 0] = smooth_image(h0_img, fwhm).get_fdata()[mask.astype("bool")]  # type: ignore
            h[:, 1] = smooth_image(h1_img, fwhm).get_fdata()[mask.astype("bool")]  # type: ignore
            h[:, 2] = smooth_image(h2_img, fwhm).get_fdata()[mask.astype("bool")]  # type: ignore

        return smooth_r1_mat, smooth_k2_mat, smooth_k2a_mat, h

    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        """Get fitted TACs based on estimated model parameters."""
        num_elements = self.tacs.num_elements
        fitted_tacs_dataobj = np.empty_like(self.tacs.dataobj)

        use_lrsc = "R1_lrsc" in self.parameters

        for i in trange(num_elements):
            idx = np.unravel_index(i, self.tacs.shape[:-1])
            dvr = self.parameters["DVR"][*idx]
            r1 = self.parameters["R1"][*idx]
            k2 = self.parameters["k2"][*idx]
            if dvr or r1 or k2:
                bp_nd = dvr - 1
                if use_lrsc:
                    r1 = self.parameters["R1_lrsc"][*idx]
                    k2 = self.parameters["k2_lrsc"][*idx]
                fitted_tacs_dataobj[*idx, :] = srtm_model(self.reftac, bp_nd, r1, k2)

        if isinstance(self.tacs, TemporalImage):
            img = image_maker(fitted_tacs_dataobj, self.tacs.img)
            ti = TemporalImage(img, self.tacs.frame_start, self.tacs.frame_duration)
            return ti
        else:
            tm = TemporalMatrix(
                fitted_tacs_dataobj, self.tacs.frame_start, self.tacs.frame_duration
            )
            return tm
