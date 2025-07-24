"""Test cases for the kineticmodel module using simulated data with ground truth."""

# ruff: noqa: S101

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray
from scipy.integrate import odeint  # type: ignore[import-untyped]

from dynamicpet.kineticmodel.logan import LRTM
from dynamicpet.kineticmodel.srtm import SRTMLammertsma1996, SRTMZhou2003
from dynamicpet.temporalobject import TemporalImage, TemporalMatrix
from dynamicpet.typing_utils import NumpyRealNumberArray

if TYPE_CHECKING:
    from nibabel.spatialimages import SpatialImage


@pytest.fixture
def frame_start() -> NDArray[np.double]:
    """Frame start times."""
    return np.array(
        [
            0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            17,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
        ],
        dtype=np.double,
    )


@pytest.fixture
def frame_duration() -> NDArray[np.double]:
    """Frame durations."""
    return np.array(
        [
            0.25,
            0.25,
            0.25,
            0.25,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            3,
            3,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
        ],
        dtype=np.double,
    )


def srtm_gen_ode_model(  # noqa: PLR0913
    y: NumpyRealNumberArray,
    _t: NumpyRealNumberArray,
    bp_nd_val: NumpyRealNumberArray,
    r1_val: NumpyRealNumberArray,
    plasma_rate: float = -0.03,
    k1prime: float = 1.0,
    k2prime: float = 0.3,
) -> list[Any]:
    """Get generative ODE for SRTM.

    Args:
        y: concentrations
        _t: time
        bp_nd_val: binding potential
        r1_val: relative radiotracer delivery
        plasma_rate: plasma rate
        k1prime: k1 for reference region
        k2prime: k2 for reference region

    Returns:
        time derivatives of concentrations

    """
    k1 = r1_val * k1prime

    dvr = 1 + bp_nd_val
    k2a = k2prime * r1_val / dvr

    cp, ct, cr = y

    dcpdt = plasma_rate * cp
    dctdt = k1 * cp - k2a * ct
    dcrdt = k1prime * cp - k2prime * cr

    return [dcpdt, dctdt, dcrdt]


def get_tacs_and_reftac_dataobj(
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
    bp_nd: float,
    r1: float,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Get digitized TACs."""
    frame_end = frame_start + frame_duration

    # initial condition
    y0 = [200, 0, 0]

    # number of time points
    num_ode_pts = 500

    t_ode = np.linspace(frame_start[0], frame_end[-1], num_ode_pts)
    y = odeint(srtm_gen_ode_model, y0, t_ode, args=(bp_nd, r1))

    # Digitize this curve
    cref = np.zeros(len(frame_start))
    ct = np.zeros(len(frame_start))
    for ti in range(len(frame_start)):
        idx = (t_ode >= frame_start[ti]) & (t_ode < frame_end[ti])
        ct[ti] = y[idx, 1].mean()
        cref[ti] = y[idx, 2].mean()

    return ct, cref


def test_srtmzhou2003_tm(
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
) -> None:
    """Test SRTM Zhou 2003 calculation using TemporalMatrix."""
    bp_nd_true = 1.5
    r1_true = 1.2
    ct, cref = get_tacs_and_reftac_dataobj(
        frame_start,
        frame_duration,
        bp_nd_true,
        r1_true,
    )

    tac = TemporalMatrix(ct, frame_start, frame_duration)
    reftac = TemporalMatrix(cref, frame_start, frame_duration)

    km = SRTMZhou2003(reftac, tac)
    # because of the way data were simulated, we need to use trapz integration
    # to fit to obtain the best possible recovery of true values
    km.fit(integration_type="trapz")

    fitted_tacs = km.fitted_tacs()

    bp_nd: NumpyRealNumberArray = km.get_parameter("BPND")  # type: ignore[assignment]
    r1: NumpyRealNumberArray = km.get_parameter("R1")  # type: ignore[assignment]

    relative_tol = 0.007  # .007 means that 0.7% error is tolerated
    assert np.allclose(bp_nd, bp_nd_true, rtol=relative_tol), "Mismatching BPND"
    assert np.allclose(r1, r1_true, rtol=relative_tol), "Mismatching R1"
    assert np.allclose(fitted_tacs.dataobj, tac.dataobj, rtol=0.01), (
        "Mismatching (fitted) TACs"
    )


def test_srtmzhou2003_ti(
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
) -> None:
    """Test SRTM Zhou 2003 calculation using TemporalImage."""
    bp_nd_true = 1.5
    r1_true = 1.2
    ct, cref = get_tacs_and_reftac_dataobj(
        frame_start,
        frame_duration,
        bp_nd_true,
        r1_true,
    )

    dims = (10, 11, 12, len(frame_start))
    img_dat = np.zeros(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                img_dat[i, j, k, :] = ct
    img = Nifti1Image(img_dat, np.eye(4))  # type: ignore[no-untyped-call]

    tac = TemporalImage(img, frame_start, frame_duration)
    reftac = TemporalMatrix(cref, frame_start, frame_duration)

    km = SRTMZhou2003(reftac, tac)
    # because of the way data were simulated, we need to use trapz integration
    # to fit to obtain the best possible recovery of true values
    km.fit(integration_type="trapz", fwhm=3)

    fitted_tacs = km.fitted_tacs()

    bp_nd: SpatialImage = km.get_parameter("BPND")  # type: ignore[assignment]
    r1: SpatialImage = km.get_parameter("R1")  # type: ignore[assignment]
    r1_lrsc: SpatialImage = km.get_parameter("R1LRSC")  # type: ignore[assignment]

    relative_tol = 0.007  # .007 means that 0.7% error is tolerated
    assert np.allclose(bp_nd.get_fdata(), bp_nd_true, rtol=relative_tol), (
        "Mismatching BPND"
    )
    assert np.allclose(r1.get_fdata(), r1_true, rtol=relative_tol), "Mismatching R1"
    assert np.allclose(r1_lrsc.get_fdata(), r1_true, rtol=relative_tol), (
        "Mismatching R1 (LRSC)"
    )
    assert np.allclose(fitted_tacs.dataobj, tac.dataobj, rtol=0.01), (
        "Mismatching (fitted) TACs"
    )


def test_srtmlammertsma1996_tm(
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
) -> None:
    """Test SRTM Lammertsma 1996 calculation using TemporalMatrix."""
    bp_nd_true = 1.5
    r1_true = 1.2
    ct, cref = get_tacs_and_reftac_dataobj(
        frame_start,
        frame_duration,
        bp_nd_true,
        r1_true,
    )

    tac = TemporalMatrix(ct, frame_start, frame_duration)
    reftac = TemporalMatrix(cref, frame_start, frame_duration)

    km = SRTMLammertsma1996(reftac, tac)
    km.fit()

    fitted_tacs = km.fitted_tacs()

    bp_nd: NumpyRealNumberArray = km.get_parameter("BPND")  # type: ignore[assignment]
    r1: NumpyRealNumberArray = km.get_parameter("R1")  # type: ignore[assignment]

    relative_tol = 0.02  # .02 means that 2% error is tolerated
    assert np.allclose(bp_nd, bp_nd_true, rtol=relative_tol), "Mismatching BPND"
    assert np.allclose(r1, r1_true, rtol=relative_tol), "Mismatching R1"
    assert np.allclose(fitted_tacs.dataobj, tac.dataobj, rtol=relative_tol), (
        "Mismatching (fitted) TACs"
    )


def test_lrtm_tm(
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
) -> None:
    """Test Logan 1996 calculation using TemporalMatrix."""
    bp_nd_true = 1.5
    r1_true = 1.2
    ct, cref = get_tacs_and_reftac_dataobj(
        frame_start,
        frame_duration,
        bp_nd_true,
        r1_true,
    )

    tac = TemporalMatrix(ct, frame_start, frame_duration)
    reftac = TemporalMatrix(cref, frame_start, frame_duration)

    km = LRTM(reftac, tac)
    # because of the way data were simulated, we need to use trapz integration
    # to fit to obtain the best possible recovery of true values
    km.fit(integration_type="trapz")

    bp_nd: NumpyRealNumberArray = km.get_parameter("BPND")  # type: ignore[assignment]

    relative_tol = 0.02  # .02 means that 2% error is tolerated
    assert np.allclose(bp_nd, bp_nd_true, rtol=relative_tol), "Mismatching BPND"
