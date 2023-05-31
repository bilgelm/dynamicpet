"""Command-line interface."""
import os
import warnings
from typing import Union

import click
from nibabel.filename_parser import splitext_addext
from nibabel.loadsave import load as nib_load
from nibabel.spatialimages import SpatialImage

from dynamicpet.denoise import hypr
from dynamicpet.kineticmodel import kinfitr
from dynamicpet.kineticmodel.srtm import SRTMLammertsma1996
from dynamicpet.kineticmodel.srtm import SRTMZhou2003
from dynamicpet.kineticmodel.suvr import SUVR
from dynamicpet.petbids import load as petbids_load
from dynamicpet.temporalobject.temporalobject import INTEGRATION_TYPE_OPTS
from dynamicpet.temporalobject.temporalobject import WEIGHT_OPTS
from dynamicpet.typing_utils import NumpyRealNumberArray


IMPLEMENTED_KMS = [
    "SUVR",
    "SRTMLammertsma1996",
    "SRTMZhou2003",
    "kinfitr.SRTM",
    "kinfitr.SRTM2",
]
IMPLEMENTED_KM_TYPES = Union[
    SUVR, SRTMLammertsma1996, SRTMZhou2003, kinfitr.SRTM, kinfitr.SRTM2
]

INTEGRATION_TYPES = (
    str(INTEGRATION_TYPE_OPTS).replace("'", "").split("[")[1].split("]")[0].split(", ")
)
WEIGHTS = str(WEIGHT_OPTS).replace("'", "").split("[")[1].split("]")[0].split(", ")


@click.group()
def denoise() -> None:
    """Denoising."""
    pass


@denoise.command()
@click.argument("pet", type=str)
@click.argument("fwhm", type=float)
@click.option(
    "--output",
    type=str,
    help=(
        "File to save denoised image. If not provided, it will be saved in the "
        "same directory as the PET image with '_hyprlr' suffix."
    ),
)
@click.option("--json", default=None, type=str, help="PET-BIDS json file")
def hypr_lr(pet: str, fwhm: float, output: str | None, json: str | None) -> None:
    """Perform HYPR-LR denoising.

    PET: 3-D or 4-D PET image

    FWHM: full width at half max, in mm, for smoothing filter
    """
    # load PET
    pet_img = petbids_load(pet, json)
    res = hypr.hypr_lr(pet_img, fwhm)
    if output is None:
        froot, ext, addext = splitext_addext(pet)
        output = froot + "_hyprlr" + ext + addext
    res.img.to_filename(output)


# ["suvr", "srtmzhou2003", "kinfitr.srtm"]
@click.command()
@click.argument("pet", type=str)
@click.argument("refmask", type=str)
@click.option(
    "--model",
    type=click.Choice(IMPLEMENTED_KMS, case_sensitive=False),
    required=True,
    help="Name of kinetic model",
)
@click.option(
    "--outputdir",
    type=str,
    help=(
        "Directory in which to save each estimated parametric image. "
        "If not provided, they will be saved in the same directory as the PET image. "
        "Outputs will have a '_<model>_<parameter>' suffix."
    ),
)
@click.option("--json", default=None, type=str, help="PET-BIDS json file")
@click.option(
    "--petmask",
    default=None,
    type=str,
    help="Binary mask specifying voxels where model should be fitted",
)
@click.option("--start", type=float, help="Start of time window for model")
@click.option("--end", type=float, help="End of time window for model")
@click.option("--fwhm", type=float, help="Full width at half max in mm for smoothing")
@click.option(
    "--weight_by",
    type=click.Choice(WEIGHTS, case_sensitive=False),
    default="frame_duration",
    help="Frame weights used in estimation procedures involving regression.",
)
@click.option(
    "--integration_type",
    type=click.Choice(INTEGRATION_TYPES, case_sensitive=False),
    default="trapz",
    help=(
        "Integration type used in estimation procedures involving "
        "time integrals. "
        "'trapz' is trapezoidal integration and "
        "'rect' is rectangular integration."
    ),
)
def kineticmodel(
    pet: str,
    refmask: str,
    model: str,
    outputdir: str | None,
    json: str | None,
    petmask: str | None,
    start: float | None,
    end: float | None,
    fwhm: float | None,
    weight_by: WEIGHT_OPTS = "frame_duration",
    integration_type: INTEGRATION_TYPE_OPTS = "trapz",
) -> None:
    """Calculate parametric images for PET image using a reference tissue model.

    PET: 3-D or 4-D PET image

    REFMASK: 3-D binary mask of reference tissue (must be in alignment with PET)
    """
    model = model.lower()

    # load PET
    pet_img = petbids_load(pet, json)
    if start is None:
        start = pet_img.start_time
    if end is None:
        end = pet_img.end_time
    if start != pet_img.start_time or end != pet_img.end_time:
        pet_img = pet_img.extract(start, end)

    refmask_img_mat: NumpyRealNumberArray = (
        nib_load(refmask).get_data().astype("bool")  # type: ignore
    )

    petmask_img_mat: NumpyRealNumberArray | None
    if petmask:
        petmask_img: SpatialImage = nib_load(petmask)  # type: ignore
        petmask_img_mat = petmask_img.get_fdata().astype("bool")
        # check that refmask is fully within petmask
        if not petmask_img_mat[refmask_img_mat].all():
            warnings.warn(
                (
                    "REFMASK is not fully within PETMASK. "
                    "Will restrict REFMASK to within PETMASK."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            refmask_img_mat = refmask_img_mat * petmask_img_mat
    else:
        petmask_img_mat = None

    # extract average TAC in reference region
    reftac = pet_img.mean_timeseries_in_mask(refmask_img_mat)

    # fit kinetic model
    km: IMPLEMENTED_KM_TYPES
    match model:
        case "suvr":
            km = SUVR(reftac, pet_img)
            km.fit(mask=petmask_img_mat)
        case "srtmlammertsma1996":
            km = SRTMLammertsma1996(reftac, pet_img)
            km.fit(weight_by=weight_by, mask=petmask_img_mat)
        case "srtmzhou2003":
            km = SRTMZhou2003(reftac, pet_img)
            km.fit(
                integration_type=integration_type,
                weight_by=weight_by,
                mask=petmask_img_mat,
                fwhm=fwhm,
            )
        case "kinfitr.srtm":
            km = kinfitr.SRTM(reftac, pet_img)
        case "kinfitr.srtm2":
            km = kinfitr.SRTM2(reftac, pet_img)
        case _:
            raise ValueError(f"Model {model} is not supported")

    froot, ext, addext = splitext_addext(pet)
    if outputdir is not None:
        bname = os.path.basename(froot)
        froot = os.path.join(outputdir, bname)

    # save estimated parameters as image
    for param in km.parameters.keys():
        res_img: SpatialImage = km.get_parameter(param)  # type: ignore
        output = froot + "_" + model.replace(".", "-") + "_" + param + ext + addext
        res_img.to_filename(output)


if __name__ == "__main__":
    kineticmodel(prog_name="kineticmodel")  # pragma: no cover
