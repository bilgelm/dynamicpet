"""Command-line interface."""
import click
from nibabel.filename_parser import splitext_addext
from nibabel.loadsave import load as nib_load
from nibabel.spatialimages import SpatialImage

from dynamicpet.kineticmodel.suvr import SUVR
from dynamicpet.petbids import load as petbids_load


@click.group()
def kineticmodel() -> None:
    """Fit a kinetic model."""
    pass


@kineticmodel.command()
@click.argument("pet", type=str)
@click.argument("refmask", type=str)
@click.option(
    "--output",
    type=str,
    help=(
        "File to save SUVR image. If not provided, it will be saved in the "
        "same directory as the PET image with '_suvr' suffix."
    ),
)
@click.option("--json", default=None, type=str, help="PET-BIDS json file")
@click.option("--start", type=float, help="Start of time window for SUVR")
@click.option("--end", type=float, help="End of time window for SUVR")
def suvr(
    pet: str,
    refmask: str,
    output: str | None,
    json: str | None,
    start: float | None,
    end: float | None,
) -> None:
    """Calculate standardized uptake value ratio (SUVR) for PET image.

    PET: 3-D or 4-D PET image

    REFMASK: 3-D binary mask of reference tissue (must be in alignment with PET)
    """
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

    # compute SUVR
    model = SUVR(reftac, pet_img)
    model.fit()

    # save SUVR as image
    suvr_img: SpatialImage = model.get_parameter("suvr")  # type: ignore
    if output is None:
        froot, ext, addext = splitext_addext(pet)
        output = froot + "_suvr" + ext + addext
    suvr_img.to_filename(output)
    pass


if __name__ == "__main__":
    kineticmodel(prog_name="kineticmodel")  # pragma: no cover
