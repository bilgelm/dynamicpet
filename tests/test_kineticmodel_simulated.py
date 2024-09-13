"""Test cases for the kineticmodel module using simulated data with ground truth."""

from typing import Any

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
from numpy.typing import NDArray
from scipy.integrate import odeint  # type: ignore

from dynamicpet.kineticmodel.logan import LRTM
from dynamicpet.kineticmodel.srtm import SRTMLammertsma1996
from dynamicpet.kineticmodel.srtm import SRTMZhou2003
from dynamicpet.temporalobject import TemporalImage
from dynamicpet.temporalobject import TemporalMatrix
from dynamicpet.typing_utils import NumpyRealNumberArray


@pytest.fixture
def frame_start() -> NDArray[np.double]:
    """Frame start times."""
    frame_start = np.array(
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
    return frame_start


@pytest.fixture
def frame_duration() -> NDArray[np.double]:
    """Frame durations."""
    frame_duration = np.array(
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
    return frame_duration


def srtm_gen_ode_model(
    y: NumpyRealNumberArray,
    t: NumpyRealNumberArray,
    bp_val: NumpyRealNumberArray,
    r1_val: NumpyRealNumberArray,
    plasma_rate: float = -0.03,
    kref1: float = 1.0,
    kref2: float = 0.3,
) -> list[Any]:
    """Generative ODE model describing SRTM.

    Args:
        y: concentrations
        t: time
        bp_val: binding potential
        r1_val: relative radiotracer delivery
        plasma_rate: plasma rate
        kref1: k1 for reference region
        kref2: k2 for reference region

    Returns:
        time derivatives of concentrations
    """
    k1 = r1_val * kref1

    dvr = 1 + bp_val
    k2a = kref2 * r1_val / dvr

    cp, ct, cr = y

    dcpdt = plasma_rate * cp
    dctdt = k1 * cp - k2a * ct
    dcrdt = kref1 * cp - kref2 * cr

    dydt = [dcpdt, dctdt, dcrdt]

    return dydt


def get_tacs_and_reftac_dataobj(
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
    bp: float,
    r1: float,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Get digitized TACs."""
    frame_end = frame_start + frame_duration

    # initial condition
    y0 = [200, 0, 0]

    # number of time points
    num_ode_pts = 500

    t_ode = np.linspace(frame_start[0], frame_end[-1], num_ode_pts)
    y = odeint(srtm_gen_ode_model, y0, t_ode, args=(bp, r1))

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
    bp_true = 1.5
    r1_true = 1.2
    ct, cref = get_tacs_and_reftac_dataobj(
        frame_start, frame_duration, bp_true, r1_true
    )

    tac = TemporalMatrix(ct, frame_start, frame_duration)
    reftac = TemporalMatrix(cref, frame_start, frame_duration)

    km = SRTMZhou2003(reftac, tac)
    # because of the way data were simulated, we need to use trapz integration
    # to fit to obtain the best possible recovery of true values
    km.fit(integration_type="trapz")

    fitted_tacs = km.fitted_tacs()

    bp: NumpyRealNumberArray = km.get_parameter("bp")  # type: ignore
    r1: NumpyRealNumberArray = km.get_parameter("r1")  # type: ignore

    relative_tol = 0.007  # .007 means that 0.7% error is tolerated
    assert np.allclose(bp, bp_true, rtol=relative_tol)
    assert np.allclose(r1, r1_true, rtol=relative_tol)
    assert np.allclose(fitted_tacs.dataobj, tac.dataobj, rtol=0.01)


def test_srtmzhou2003_ti(
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
) -> None:
    """Test SRTM Zhou 2003 calculation using TemporalImage."""
    bp_true = 1.5
    r1_true = 1.2
    ct, cref = get_tacs_and_reftac_dataobj(
        frame_start, frame_duration, bp_true, r1_true
    )

    dims = (10, 11, 12, len(frame_start))
    img_dat = np.zeros(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                img_dat[i, j, k, :] = ct
    img = Nifti1Image(img_dat, np.eye(4))  # type: ignore

    tac = TemporalImage(img, frame_start, frame_duration)
    reftac = TemporalMatrix(cref, frame_start, frame_duration)

    km = SRTMZhou2003(reftac, tac)
    # because of the way data were simulated, we need to use trapz integration
    # to fit to obtain the best possible recovery of true values
    km.fit(integration_type="trapz", fwhm=3)

    fitted_tacs = km.fitted_tacs()

    bp: SpatialImage = km.get_parameter("bp")  # type: ignore
    r1: SpatialImage = km.get_parameter("r1")  # type: ignore
    r1_lrsc: SpatialImage = km.get_parameter("r1_lrsc")  # type: ignore

    relative_tol = 0.007  # .007 means that 0.7% error is tolerated
    assert np.allclose(bp.get_fdata(), bp_true, rtol=relative_tol)
    assert np.allclose(r1.get_fdata(), r1_true, rtol=relative_tol)
    assert np.allclose(r1_lrsc.get_fdata(), r1_true, rtol=relative_tol)
    assert np.allclose(fitted_tacs.dataobj, tac.dataobj, rtol=0.01)


def test_srtmlammertsma1996_tm(
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
) -> None:
    """Test SRTM Lammertsma 1996 calculation using TemporalMatrix."""
    bp_true = 1.5
    r1_true = 1.2
    ct, cref = get_tacs_and_reftac_dataobj(
        frame_start, frame_duration, bp_true, r1_true
    )

    tac = TemporalMatrix(ct, frame_start, frame_duration)
    reftac = TemporalMatrix(cref, frame_start, frame_duration)

    km = SRTMLammertsma1996(reftac, tac)
    km.fit()

    fitted_tacs = km.fitted_tacs()

    bp: NumpyRealNumberArray = km.get_parameter("bp")  # type: ignore
    r1: NumpyRealNumberArray = km.get_parameter("r1")  # type: ignore

    relative_tol = 0.02  # .02 means that 2% error is tolerated
    assert np.allclose(bp, bp_true, rtol=relative_tol)
    assert np.allclose(r1, r1_true, rtol=relative_tol)
    assert np.allclose(fitted_tacs.dataobj, tac.dataobj, rtol=relative_tol)


def test_lrtm_tm(
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
) -> None:
    """Test Logan 1996 calculation using TemporalMatrix."""
    bp_true = 1.5
    r1_true = 1.2
    ct, cref = get_tacs_and_reftac_dataobj(
        frame_start, frame_duration, bp_true, r1_true
    )

    tac = TemporalMatrix(ct, frame_start, frame_duration)
    reftac = TemporalMatrix(cref, frame_start, frame_duration)

    km = LRTM(reftac, tac)
    # because of the way data were simulated, we need to use trapz integration
    # to fit to obtain the best possible recovery of true values
    km.fit(integration_type="trapz")

    bp: NumpyRealNumberArray = km.get_parameter("bp")  # type: ignore

    relative_tol = 0.02  # .02 means that 2% error is tolerated
    assert np.allclose(bp, bp_true, rtol=relative_tol)
