"""Test cases for the kinfitr wrapper using data provided in kinfitr."""
from collections import namedtuple
import numpy as np
import pytest

from rpy2.robjects.packages import data as rdata  # type: ignore
from rpy2.robjects.packages import importr

from dynamicpet.kineticmodel import kinfitr
from dynamicpet.kineticmodel.srtm import SRTMLammertsma1996
from dynamicpet.kineticmodel.srtm import SRTMZhou2003
from dynamicpet.temporalobject import TemporalMatrix

TACPair = namedtuple('TACPair', ['reftac', 'tacs'])


@pytest.fixture
def simref0() -> TACPair:
    """Get data for first subject in kinfitr's simref dataset."""
    kinfitr = importr('kinfitr')
    simref = rdata(kinfitr).fetch('simref')['simref']
    # get data for participant at index i
    i = 0
    simref_i = np.array(simref[3][i])

    # Times, Reference, ROI1, ROI2, ROI3, Weights, StartTime, Duration
    _, reftac, tac1, tac2, tac3, weights, frame_start, frame_duration = simref_i

    frame_start = frame_start
    frame_duration = frame_duration

    # mitigate overlap issues by manipulating frame_duration
    frame_duration = np.append(frame_start[1:] - frame_start[:-1], frame_duration[-1])

    print(frame_start)
    print(frame_duration)
    frame_end = frame_start + frame_duration
    print(frame_end[:-1] - frame_start[1:])

    # drop first frame (which has 0 duration, so creates TemporalMatrix problems)
    # this should not affect kinfitr functions as they will add the 0 back
    reftac_tm = TemporalMatrix(reftac[1:], frame_start[1:], frame_duration[1:])
    tac_tm = TemporalMatrix(np.row_stack((tac1[1:], tac2[1:], tac3[1:])),
                            frame_start[1:], frame_duration[1:])

    return TACPair(reftac_tm, tac_tm)


def test_kinfitr_srtm(simref0: TACPair) -> None:
    """Test kinfitr SRTM wrapper."""
    km = kinfitr.SRTM(simref0.reftac, simref0.tacs)
    km.fit()

    # check that the results match those provided in kinfitr documentation
    bp: float = km.get_parameter("bp")[0]  # type: ignore
    r1: float = km.get_parameter("R1")[0]  # type: ignore
    k2: float = km.get_parameter("k2")[0]  # type: ignore

    assert np.round(bp, 6) == 1.488339
    assert np.round(r1, 6) == 1.233546
    assert np.round(k2, 6) == 0.101624


def test_kinfitr_srtm2(simref0: TACPair) -> None:
    """Test kinfitr SRTM2 wrapper."""
    km = kinfitr.SRTM2(simref0.reftac, simref0.tacs)
    km.fit()

    km.get_parameter("bp")
    km.get_parameter("k2a")


# def test_kinfitr_frtm(simref0: TACPair) -> None:
#     """Test kinfitr SRTM2 wrapper."""
#     km = kinfitr.FRTM(simref0.reftac, simref0.tacs)
#     km.fit()

#     km.get_parameter("bp")
#     km.get_parameter("k3")


def test_kinfitr_mrtm1(simref0: TACPair) -> None:
    """Test kinfitr MRTM1 wrapper using TemporalMatrix."""
    km = kinfitr.MRTM1(simref0.reftac, simref0.tacs)
    km.fit(frameStartEnd=np.array([1, 20]))

    print(km.get_parameter("bp"))
    print(km.get_parameter("k2prime"))


def test_kinfitr_mrtm2(simref0: TACPair) -> None:
    """Test kinfitr MRTM2 wrapper."""
    km = kinfitr.MRTM2(simref0.reftac, simref0.tacs)
    km.fit(k2prime=.1, frameStartEnd=np.array([1, 20]))

    # check that the results match those provided in kinfitr documentation
    bp: float = km.get_parameter("bp")[0]  # type: ignore
    k2: float = km.get_parameter("k2")[0]  # type: ignore

    print(bp)
    print(k2)

    # assert np.round(bp, 2) == round(1.488339, 2)


# def test_kinfitr_reflogan(simref0: TACPair) -> None:
#     """Test kinfitr reflogan. DOESNT WORK; rpy2 error"""
#     km = kinfitr.RefLogan(simref0.reftac, simref0.tacs)
#     km.fit(k2prime=.1, tstarIncludedFrames=15)

#     print(km.get_parameter("bp"))


def test_srtm_zhou2003(simref0: TACPair) -> None:
    """Test SRTM Zhou 2003."""
    km = SRTMZhou2003(simref0.reftac, simref0.tacs)
    km.fit()

    # check that the results match those provided in kinfitr documentation
    bp: float = km.get_parameter("bp")[0]  # type: ignore
    r1: float = km.get_parameter("r1")[0]  # type: ignore
    k2: float = km.get_parameter("k2")[0]  # type: ignore

    print(bp)
    print(r1)
    print(k2)

    assert np.round(bp, 2) == round(1.488339, 2)
    assert np.round(r1, 1) == round(1.233546, 1)
    assert np.round(k2, 1) == round(0.101624, 1)


def test_srtm_zhou2003_trapz(simref0: TACPair) -> None:
    """Test SRTM Zhou 2003."""
    km = SRTMZhou2003(simref0.reftac, simref0.tacs)
    km.fit(integration_type='trapz')

    # check that the results match those provided in kinfitr documentation
    bp: float = km.get_parameter("bp")[0]  # type: ignore
    r1: float = km.get_parameter("r1")[0]  # type: ignore
    k2: float = km.get_parameter("k2")[0]  # type: ignore

    print(bp)
    print(r1)
    print(k2)

    assert np.round(bp, 1) == round(1.488339, 1)
    assert np.round(r1, 1) == round(1.233546, 1)
    assert np.round(k2, 1) == round(0.101624, 1)


def test_srtm_lammertsma1996(simref0: TACPair) -> None:
    """Test SRTM Lammertsma 1996."""
    km = SRTMLammertsma1996(simref0.reftac, simref0.tacs)
    km.fit()

    # check that the results match those provided in kinfitr documentation
    bp: float = km.get_parameter("bp")[0]  # type: ignore
    r1: float = km.get_parameter("r1")[0]  # type: ignore
    k2: float = km.get_parameter("k2")[0]  # type: ignore

    print(bp)
    print(r1)
    print(k2)

    assert np.round(bp, 1) == round(1.488339, 1)
    assert np.round(r1, 1) == round(1.233546, 1)
    assert np.round(k2, 2) == round(0.101624, 2)
