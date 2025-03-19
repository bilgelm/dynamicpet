"""Test cases for __main__.py."""

# ruff: noqa: S101

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import requests  # consider revisiting betamax after urllib3 bug is fixed
from click.testing import CliRunner
from nibabel.loadsave import load as nib_load

from dynamicpet.temporalobject.temporalimage import image_maker

if TYPE_CHECKING:
    from nibabel.spatialimages import SpatialImage


@pytest.fixture(scope="session")
def images() -> dict[str, Path]:
    """Download test files from OpenNeuro."""
    outdir = Path(__file__).parent / "test_data"
    outdir.mkdir(exist_ok=True)

    rois_fname = outdir / "sub-000101_ses-baseline_label-displacementROI_dseg.nii.gz"
    petjson_fname = outdir / "sub-000101_ses-baseline_pet.json"
    pet_fname = outdir / "sub-000101_ses-baseline_pet.nii"

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
        with Path(rois_fname).open("wb") as f:
            f.write(r.content)

    if not petjson_fname.exists():
        r = requests.get(
            baseurl
            + "pet/sub-000101_ses-baseline_pet.json"
            + "?versionId=Gfkc8Y71JexOLZq40ZN4BTln_4VObTJR",
            timeout=10,
        )
        r.raise_for_status()
        with Path(petjson_fname).open("wb") as f:
            f.write(r.content)

    if not pet_fname.exists():
        with requests.get(peturl, timeout=10, stream=True) as r:
            r.raise_for_status()
            with Path(pet_fname).open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    petmask_fname = outdir / "sub-000101_ses-baseline_desc-pet_mask.nii.gz"
    refmask_fname = outdir / "sub-000101_ses-baseline_desc-ref_mask.nii.gz"
    rois_csv_fname = outdir / "sub-000101_ses-baseline_tacs.tsv"

    if petmask_fname.exists() and refmask_fname.exists() and rois_csv_fname.exists():
        pass
    else:
        # save timeseries in each ROI to a single csv file
        pet_img: SpatialImage = nib_load(pet_fname)  # type: ignore[assignment]
        pet_data = pet_img.get_fdata()

        rois_img: SpatialImage = nib_load(rois_fname)  # type: ignore[assignment]
        rois_data = rois_img.get_fdata()
        num_rois = len(np.unique(rois_data)) - 1

        rois_ts = np.empty((num_rois, pet_img.shape[-1]))
        for i in range(num_rois):
            rois_ts[i, :] = pet_data[rois_data == i + 1, :].mean(axis=0)

        thresh = 100
        petmask_img = image_maker(pet_data.sum(axis=-1) > thresh, pet_img)
        petmask_img.to_filename(petmask_fname)

        # make a binary mask from first ROI in displacementROI_dseg.nii.gz
        # as a quick hack, we use petmask_img header info here rather than rois_img
        # to ensure that the two masks are in the same space
        refmask_img = image_maker(1 * (rois_data == 1), petmask_img)
        refmask_img.to_filename(refmask_fname)

        with Path(rois_csv_fname).open("w") as f:
            tsvwriter = csv.writer(f, delimiter="\t")
            # write ROI names in first row
            tsvwriter.writerow(["ROI" + str(i + 1) for i in range(num_rois)])
            for row in rois_ts.T:
                tsvwriter.writerow(row)

    return {
        "pet_fname": pet_fname,
        "petjson_fname": petjson_fname,
        "petmask_fname": petmask_fname,
        "refmask_fname": refmask_fname,
        "rois_fname": rois_fname,
        "rois_csv_fname": rois_csv_fname,
    }


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

    assert result.exit_code == 0, "Denoise had an error"

    msg = "Couldn't find denoise output file"
    assert (outputdir / "sub-000101_ses-baseline_desc-hyprlr_pet.nii").is_file(), msg
    assert (outputdir / "sub-000101_ses-baseline_desc-hyprlr_pet.json").is_file(), msg


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

    assert result.exit_code == 0, "Denoise had an error"

    msg = "Couldn't find denoise output file"
    assert (outputdir / "sub-000101_ses-baseline_desc-nesma_pet.nii").is_file(), msg
    assert (outputdir / "sub-000101_ses-baseline_desc-nesma_pet.json").is_file(), msg


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
    assert result.exit_code == 0, "Kineticmodel had an error"

    msg = "Couldn't find kineticmodel output file"
    assert (outputdir / "sub-000101_ses-baseline_model-SUVR_kinpar.tsv").is_file(), msg

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
    assert result.exit_code == 0, "Kineticmodel had an error"

    msg = "Couldn't find kineticmodel output file"
    assert (
        outputdir / "sub-000101_ses-baseline_model-SUVR_meas-SUVR_mimap.nii"
    ).is_file(), msg
    assert (outputdir / "sub-000101_ses-baseline_model-SUVR_mimap.json").is_file(), msg

    # finally, make sure that the two methods give the same results

    # skip the header (column names) and the first column (ROI names)
    rois_km_suvr = np.genfromtxt(
        outputdir / "sub-000101_ses-baseline_model-SUVR_kinpar.tsv",
        delimiter="\t",
        skip_header=1,
        usecols=(1,),
    )

    # read in rois_fname
    rois_img: SpatialImage = nib_load(images["rois_fname"])  # type: ignore[assignment]

    # calculate avg SUVR (pet_model-SUVR_meas-SUVR_mimap.nii) per ROI in rois_img
    pet_km_suvr_img: SpatialImage = nib_load(  # type: ignore[assignment]
        outputdir / "sub-000101_ses-baseline_model-SUVR_meas-SUVR_mimap.nii",
    )
    pet_km_suvr = pet_km_suvr_img.get_fdata()
    num_rois = len(np.unique(rois_img.get_fdata())) - 1
    suvr = np.empty(num_rois)
    for i in range(num_rois):
        suvr[i] = pet_km_suvr[rois_img.get_fdata() == i + 1].mean()

    # check that the suvr of the reference region is 1.0
    assert np.round(suvr[0], 4) == 1.0, (
        f"Expected SUVR of 1.0 in reference, got {np.round(suvr[0], 4)}"
    )
    assert np.allclose(rois_km_suvr, suvr, rtol=1e-3, atol=1e-3), "Mismatching SUVRs"


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
    assert result.exit_code == 0, "Kineticmodel had an error"

    msg = "Couldn't find kineticmodel output file"
    assert (outputdir / "sub-000101_ses-baseline_model-SRTM_kinpar.tsv").is_file(), msg
    assert (outputdir / "sub-000101_ses-baseline_model-SRTM_kinpar.json").is_file(), msg

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
    assert result.exit_code == 0, "Kineticmodel had an error"

    msg = "Couldn't find kineticmodel output file"
    assert (
        outputdir / "sub-000101_ses-baseline_model-SRTM_meas-DVR_mimap.nii"
    ).is_file(), msg
    assert (
        outputdir / "sub-000101_ses-baseline_model-SRTM_meas-R1_mimap.nii"
    ).is_file(), msg
    assert (outputdir / "sub-000101_ses-baseline_model-SRTM_mimap.json").is_file(), msg
