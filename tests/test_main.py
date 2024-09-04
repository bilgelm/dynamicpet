"""Test cases for __main__.py."""

import csv
import os
from pathlib import Path

import numpy as np
import pytest

# from betamax import Betamax
import requests
from click.testing import CliRunner
from nibabel.loadsave import load as nib_load
from nibabel.spatialimages import SpatialImage

from dynamicpet.temporalobject.temporalimage import image_maker


# pytest.mark.usefixtures("betamax_session")

# outdir = Path(__file__).parent / "test_data"
# outdir.mkdir(exist_ok=True)
# with Betamax.configure() as config:
#     config.cassette_library_dir = outdir
#
# will revisit betamax after urllib3 bug is fixed


@pytest.fixture(scope="session")
def images() -> dict[str, Path]:
    """Download test files from OpenNeuro."""
    outdir = Path(__file__).parent / "test_data"
    outdir.mkdir(exist_ok=True)

    rois_fname = outdir / "displacementROI_dseg.nii.gz"
    petjson_fname = outdir / "pet.json"
    pet_fname = outdir / "pet.nii"

    baseurl = "https://s3.amazonaws.com/openneuro.org/ds001705/sub-000101/ses-baseline/"
    roisurl = (
        baseurl
        + "anat/sub-000101_ses-baseline_label-displacementROI_dseg.nii.gz"
        + "?versionId=B_o6d4NjZsRkhDaJMwZ8GFnxjk72H6e1"
    )
    peturl = (
        baseurl
        + "pet/sub-000101_ses-baseline_pet.nii"
        + "?versionId=rMjWUWxAIYI46DmOQjulNQLTDUAThT5o"
    )

    if not rois_fname.exists():
        r = requests.get(roisurl, timeout=10)
        r.raise_for_status()
        with open(rois_fname, "wb") as f:
            f.write(r.content)

    if not petjson_fname.exists():
        r = requests.get(
            baseurl
            + "pet/sub-000101_ses-baseline_pet.json"
            + "?versionId=Gfkc8Y71JexOLZq40ZN4BTln_4VObTJR",
            timeout=10,
        )
        r.raise_for_status()
        with open(petjson_fname, "wb") as f:
            f.write(r.content)

    if not pet_fname.exists():
        with requests.get(peturl, timeout=10, stream=True) as r:
            r.raise_for_status()
            with open(pet_fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    petmask_fname = outdir / "petmask.nii.gz"
    refmask_fname = outdir / "refmask.nii.gz"
    rois_csv_fname = outdir / "rois.tsv"

    if petmask_fname.exists() and refmask_fname.exists() and rois_csv_fname.exists():
        pass
    else:
        # save timeseries in each ROI to a single csv file
        pet_img: SpatialImage = nib_load(pet_fname)  # type: ignore
        pet_data = pet_img.get_fdata()

        rois_img: SpatialImage = nib_load(rois_fname)  # type: ignore
        rois_data = rois_img.get_fdata()
        num_rois = len(np.unique(rois_data)) - 1

        rois_ts = np.empty((num_rois, pet_img.shape[-1]))
        for i in range(num_rois):
            rois_ts[i, :] = pet_data[rois_data == i + 1, :].mean(axis=0)

        petmask_img = image_maker(pet_data.sum(axis=-1) > 100, pet_img)
        petmask_img.to_filename(petmask_fname)

        # make a binary mask from first ROI in displacementROI_dseg.nii.gz
        # as a quick hack, we use petmask_img header info here rather than rois_img
        # to ensure that the two masks are in the same space
        refmask_img = image_maker(1 * (rois_data == 1), petmask_img)
        refmask_img.to_filename(refmask_fname)

        with open(rois_csv_fname, "w") as f:
            tsvwriter = csv.writer(f, delimiter="\t")
            # write ROI names in first row
            tsvwriter.writerow(["ROI" + str(i + 1) for i in range(num_rois)])
            for row in rois_ts.T:
                tsvwriter.writerow(row)

    fnames = {
        "pet_fname": pet_fname,
        "petjson_fname": petjson_fname,
        "petmask_fname": petmask_fname,
        "refmask_fname": refmask_fname,
        "rois_fname": rois_fname,
        "rois_csv_fname": rois_csv_fname,
    }

    return fnames


def test_denoise_hyprlr(images: dict[str, Path]) -> None:
    """Test denoise in __main__.py."""
    from dynamicpet.__main__ import denoise

    pet_fname = images["pet_fname"]
    outputdir = pet_fname.parent / "test_output" / "hyprlr"

    runner = CliRunner()
    result = runner.invoke(
        denoise,
        [
            "--method",
            "HYPRLR",
            str(pet_fname),
            "--fwhm",
            "5.0",
            "--outputdir",
            str(outputdir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert os.path.isfile(outputdir / "pet_hyprlr.nii")
    assert os.path.isfile(outputdir / "pet_hyprlr.json")


def test_denoise_nesma(images: dict[str, Path]) -> None:
    """Test denoise in __main__.py."""
    from dynamicpet.__main__ import denoise

    pet_fname = images["pet_fname"]
    petmask_fname = images["petmask_fname"]
    outputdir = pet_fname.parent / "test_output" / "nesma"

    runner = CliRunner()
    result = runner.invoke(
        denoise,
        [
            "--method",
            "NESMA",
            str(pet_fname),
            "--mask",
            str(petmask_fname),
            "--window_half_size",
            "3",
            "3",
            "3",
            "--thresh",
            "0.05",
            "--outputdir",
            str(outputdir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert os.path.isfile(outputdir / "pet_nesma.nii")
    assert os.path.isfile(outputdir / "pet_nesma.json")


def test_kineticmodel_suvr(images: dict[str, Path]) -> None:
    """Test SUVR kineticmodel in __main__.py."""
    from dynamicpet.__main__ import kineticmodel

    # first, test with tsv TACs
    petjson_fname = images["petjson_fname"]
    rois_csv_fname = images["rois_csv_fname"]
    outputdir = rois_csv_fname.parent / "test_output" / "suvr"

    runner = CliRunner()
    result = runner.invoke(
        kineticmodel,
        [
            str(rois_csv_fname),
            "--model",
            "suvr",
            "--refroi",
            "ROI1",
            "--json",
            str(petjson_fname),
            "--start",
            str(50),
            "--end",
            str(70),
            "--outputdir",
            str(outputdir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert (outputdir / "rois_km-suvr.tsv").is_file()

    # next, test with nifti images
    pet_fname = images["pet_fname"]
    petmask_fname = images["petmask_fname"]
    refmask_fname = images["refmask_fname"]
    outputdir = pet_fname.parent / "test_output" / "suvr"

    runner = CliRunner()
    result = runner.invoke(
        kineticmodel,
        [
            str(pet_fname),
            "--model",
            "suvr",
            "--refmask",
            str(refmask_fname),
            "--petmask",
            str(petmask_fname),
            "--start",
            str(50),
            "--end",
            str(70),
            "--outputdir",
            str(outputdir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert os.path.isfile(outputdir / "pet_km-suvr_kp-suvr.nii")

    # finally, make sure that the two methods give the same results

    # skip the header (column names) and the first column (ROI names)
    rois_km_suvr = np.genfromtxt(
        outputdir / "rois_km-suvr.tsv", delimiter="\t", skip_header=1, usecols=(1,)
    )

    # read in rois_fname
    rois_img: SpatialImage = nib_load(images["rois_fname"])  # type: ignore

    # calculate the average SUVR (pet_km-suvr_kp-suvr.nii) in each ROI in rois_img
    pet_km_suvr_img: SpatialImage = nib_load(  # type: ignore
        outputdir / "pet_km-suvr_kp-suvr.nii"
    )
    pet_km_suvr = pet_km_suvr_img.get_fdata()
    num_rois = len(np.unique(rois_img.get_fdata())) - 1
    suvr = np.empty(num_rois)
    for i in range(num_rois):
        suvr[i] = pet_km_suvr[rois_img.get_fdata() == i + 1].mean()

    # check that the suvr of the reference region is 1.0
    assert np.round(suvr[0], 4) == 1.0
    assert np.allclose(rois_km_suvr, suvr, rtol=1e-3, atol=1e-3)


def test_kineticmodel_srtmzhou2003(images: dict[str, Path]) -> None:
    """Test SRTM Zhou 2003 kineticmodel in __main__.py."""
    from dynamicpet.__main__ import kineticmodel

    # first, test with tsv TACs
    petjson_fname = images["petjson_fname"]
    rois_csv_fname = images["rois_csv_fname"]
    outputdir = rois_csv_fname.parent / "test_output" / "srtmzhou2003"

    model = "srtmzhou2003"

    runner = CliRunner()
    result = runner.invoke(
        kineticmodel,
        [
            str(rois_csv_fname),
            "--model",
            model,
            "--refroi",
            "ROI1",
            "--json",
            str(petjson_fname),
            "--start",
            str(0),
            "--end",
            str(70),
            "--outputdir",
            str(outputdir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert os.path.isfile(outputdir / ("rois_km-" + model + ".tsv"))

    # next, test with nifti images
    pet_fname = images["pet_fname"]
    petmask_fname = images["petmask_fname"]
    refmask_fname = images["refmask_fname"]
    outputdir = pet_fname.parent / "test_output" / "srtmzhou2003"

    runner = CliRunner()
    result = runner.invoke(
        kineticmodel,
        [
            str(pet_fname),
            "--model",
            model,
            "--refmask",
            str(refmask_fname),
            "--petmask",
            str(petmask_fname),
            "--start",
            str(0),
            "--end",
            str(70),
            "--outputdir",
            str(outputdir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert os.path.isfile(outputdir / ("pet_km-" + model + "_kp-dvr.nii"))
    assert os.path.isfile(outputdir / ("pet_km-" + model + "_kp-r1.nii"))


def test_kineticmodel_kinfitr_srtm(images: dict[str, Path]) -> None:
    """Test kinfitr's SRTM in __main__.py."""
    pytest.importorskip("dynamicpet.kineticmodel.kinfitr")

    from dynamicpet.__main__ import kineticmodel

    # first, test with tsv TACs
    petjson_fname = images["petjson_fname"]
    rois_csv_fname = images["rois_csv_fname"]
    outputdir = rois_csv_fname.parent / "test_output" / "kinfitr_srtm"

    model = "kinfitr.srtm"

    runner = CliRunner()
    result = runner.invoke(
        kineticmodel,
        [
            str(rois_csv_fname),
            "--model",
            model,
            "--refroi",
            "ROI1",
            "--json",
            str(petjson_fname),
            "--start",
            str(0),
            "--end",
            str(70),
            "--outputdir",
            str(outputdir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert os.path.isfile(outputdir / ("rois_km-" + model.replace(".", "") + ".tsv"))
