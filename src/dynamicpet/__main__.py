"""Command-line interface."""
import click
import os
from nibabel.filename_parser import splitext_addext
from nibabel.loadsave import load as nib_load
from nibabel.spatialimages import SpatialImage

from dynamicpet.kineticmodel.suvr import SUVR
from dynamicpet.petbids import load as petbids_load
from dynamicpet.denoise import hypr


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
        "File to save SUVR image. If not provided, it will be saved in the "
        "same directory as the PET image with '_hyprlr' suffix."
    ),
)
@click.option("--json", default=None, type=str, help="PET-BIDS json file")
def hypr_lr(
    pet: str,
    fwhm: float,
    output: str | None,
    json: str | None
) -> None:
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


@click.group()
def kineticmodel() -> None:
    """Fit a kinetic model."""
    pass


@kineticmodel.command()
@click.argument("pet", type=str)
@click.argument("refmask", type=str)
@click.option(
    "--outputdir",
    type=str,
    help=(
        "Directory in which to save SUVR image. If not provided, it will be saved "
        "in the same directory as the PET image. Output will have a '_suvr' suffix."
    ),
)
@click.option("--json", default=None, type=str, help="PET-BIDS json file")
@click.option("--petmask", default=None, type=str,
              help="Binary mask specifying voxels where SUVR should be calculated")
@click.option("--start", type=float, help="Start of time window for SUVR")
@click.option("--end", type=float, help="End of time window for SUVR")
def suvr(
    pet: str,
    refmask: str,
    outputdir: str | None,
    json: str | None,
    petmask: str | None,
    start: float | None,
    end: float | None,
) -> None:
    """Calculate standardized uptake value ratio (SUVR) for PET image.

    PET: 3-D or 4-D PET image

    REFMASK: 3-D binary mask of reference tissue (must be in alignment with PET)
    """
    cls = SUVR

    # load PET
    pet_img = petbids_load(pet, json)
    if start is None:
        start = pet_img.start_time
    if end is None:
        end = pet_img.end_time
    pet_img = pet_img.extract(start, end)

    # extract average reference region TAC
    refmask_img: SpatialImage = nib_load(refmask)  # type: ignore
    # refmask_img = SpatialImage.from_filename(refmask)
    reftac = pet_img.mean_timeseries_in_mask(refmask_img)

    if petmask:
        petmask_img: SpatialImage = nib_load(petmask)  # type: ignore
    else:
        petmask_img = None

    # compute SUVR
    model = cls(reftac, pet_img)
    model.fit(mask=petmask_img)

    froot, ext, addext = splitext_addext(pet)
    if outputdir is not None:
        bname = os.path.basename(froot)
        froot = os.path.join(outputdir, bname)

    # save SUVR as image
    for param in model.parameters.keys():
        res_img: SpatialImage = model.get_parameter(param)  # type: ignore
        output = froot + "_" + param + ext + addext
        res_img.to_filename(output)


if __name__ == "__main__":
    kineticmodel(prog_name="kineticmodel")  # pragma: no cover
