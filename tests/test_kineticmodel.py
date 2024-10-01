"""Test cases for the kineticmodel module."""

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
from numpy.typing import NDArray

from dynamicpet.kineticmodel.logan import LRTM
from dynamicpet.kineticmodel.srtm import SRTMZhou2003
from dynamicpet.kineticmodel.suvr import SUVR
from dynamicpet.temporalobject import TemporalImage
from dynamicpet.temporalobject import TemporalMatrix
from dynamicpet.typing_utils import NumpyRealNumberArray


@pytest.fixture
def tacs_dataobj() -> NDArray[np.double]:
    """Time activity curves for two regions."""
    return np.array([[60, 60, 60, 60], [90, 90, 90, 90]], dtype=np.double)


@pytest.fixture
def reftac_dataobj() -> NDArray[np.double]:
    """Reference TAC."""
    return np.array([[30, 30, 30, 30]], dtype=np.double)


@pytest.fixture
def frame_start() -> NDArray[np.double]:
    """Frame start times."""
    return np.array([0, 5, 10, 20], dtype=np.double) * 60


@pytest.fixture
def frame_duration() -> NDArray[np.double]:
    """Frame durations."""
    return np.array([5, 5, 10, 10], dtype=np.double) * 60


@pytest.fixture
def reftac(
    reftac_dataobj: NDArray[np.double],
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
) -> TemporalMatrix:
    """Create a TemporalMatrix for reference TAC."""
    return TemporalMatrix(reftac_dataobj, frame_start, frame_duration)


@pytest.fixture
def tacs(
    tacs_dataobj: NDArray[np.double],
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
) -> TemporalMatrix:
    """Create a TemporalMatrix for reference TAC."""
    return TemporalMatrix(tacs_dataobj, frame_start, frame_duration)


@pytest.fixture
def tacs_img(
    tacs_dataobj: NDArray[np.double],
    frame_start: NDArray[np.double],
    frame_duration: NDArray[np.double],
) -> TemporalImage:
    """Create a TemporalMatrix for reference TAC."""
    tacs_as_img = Nifti1Image(  # type: ignore
        tacs_dataobj[np.newaxis, np.newaxis, :, :], np.eye(4)
    )
    return TemporalImage(tacs_as_img, frame_start, frame_duration)


def test_suvr_tm(reftac: TemporalMatrix, tacs: TemporalMatrix) -> None:
    """Test SUVR calculation using TemporalMatrix."""
    km = SUVR(reftac, tacs)
    km.fit()
    suvr: NumpyRealNumberArray = km.get_parameter("SUVR")  # type: ignore
    assert suvr.shape == (2,)
    assert np.all(suvr == [2, 3])


def test_suvr_ti(reftac: TemporalMatrix, tacs_img: TemporalImage) -> None:
    """Test SUVR calculation using TemporalImage."""
    km = SUVR(reftac, tacs_img)
    km.fit()

    fitted_tacs = km.fitted_tacs()

    suvr_img: SpatialImage = km.get_parameter("SUVR")  # type: ignore
    assert suvr_img.shape == (1, 1, 2)
    assert np.all(suvr_img.get_fdata() == np.array([[[2, 3]]]))
    assert np.all(fitted_tacs.dataobj == tacs_img.dataobj)


def test_srtm_zhou2003_ti(reftac: TemporalMatrix, tacs_img: TemporalImage) -> None:
    """Test SRTM Zhou 2003 using TemporalImage."""
    km = SRTMZhou2003(reftac, tacs_img)
    km.fit(integration_type="trapz")
    dvr_img: SpatialImage = km.get_parameter("DVR")  # type: ignore

    assert dvr_img.shape == (1, 1, 2)

    bp_nd_img: SpatialImage = km.get_parameter("BPND")  # type: ignore

    assert np.allclose(dvr_img.get_fdata(), bp_nd_img.get_fdata() + 1)


def test_logan_tm(reftac: TemporalMatrix, tacs_img: TemporalImage) -> None:
    """Test Logan Plot using TemporalImage."""
    km = LRTM(reftac, tacs_img)
    km.fit(integration_type="trapz")

    dvr_img: SpatialImage = km.get_parameter("DVR")  # type: ignore
    assert dvr_img.shape == (1, 1, 2)
