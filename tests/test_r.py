"""Test system install of R."""
from subprocess import run


def test_r_version() -> None:
    """Test if system install of R can be accessed."""
    run(["R", "--version"])


def test_kinfitr_load() -> None:
    """Test if kinfitr can be loaded in R."""
    run(["Rscript", "-e", "'library(kinfitr)'"])


def test_rpy2_import() -> None:
    """Test if rpy2 python module can be imported."""
    import rpy2  # type: ignore
