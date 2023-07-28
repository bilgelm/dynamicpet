"""Command-line interface."""
import csv
import os
import warnings
from typing import Tuple
from typing import Union

import click
import numpy as np
from nibabel.filename_parser import splitext_addext
from nibabel.loadsave import load as nib_load
from nibabel.spatialimages import SpatialImage

from dynamicpet.denoise import hypr
from dynamicpet.kineticmodel import kinfitr
from dynamicpet.kineticmodel.srtm import SRTMLammertsma1996
from dynamicpet.kineticmodel.srtm import SRTMZhou2003
from dynamicpet.kineticmodel.suvr import SUVR
from dynamicpet.petbids import PETBIDSImage
from dynamicpet.petbids import PETBIDSMatrix
from dynamicpet.petbids.petbidsimage import load as petbidsimage_load
from dynamicpet.petbids.petbidsmatrix import load as petbidsmatrix_load
from dynamicpet.temporalobject import TemporalMatrix
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


@click.command()
@click.argument("pet", type=str)
@click.argument("fwhm", type=float)
@click.option(
    "--method",
    type=click.Choice(["HYPRLR"], case_sensitive=False),
    required=True,
    help="Name of denoising method",
)
@click.option(
    "--output",
    default=None,
    type=str,
    help=(
        "File to save denoised image. If not provided, it will be saved in the "
        "same directory as the PET image with '_<method>' suffix."
    ),
)
@click.option("--json", default=None, type=str, help="PET-BIDS json file")
def denoise(
    pet: str, fwhm: float, method: str, output: str | None, json: str | None
) -> None:
    """Perform dynamic PET denoising.

    PET: 4-D PET image

    FWHM: full width at half max, in mm, for smoothing filter
    """
    # load PET
    pet_img = petbidsimage_load(pet, json)

    if method == "HYPRLR":
        res = hypr.hypr_lr(pet_img, fwhm)
    else:
        raise NotImplementedError(f"Denoising method {method} is not supported")

    if output is None:
        froot, ext, addext = splitext_addext(pet)
        output = froot + "_hyprlr" + ext + addext
    res.to_filename(output)


@click.command()
@click.argument("pet", type=str)
@click.option(
    "--model",
    type=click.Choice(IMPLEMENTED_KMS, case_sensitive=False),
    required=True,
    help="Name of kinetic model",
)
@click.option(
    "--refroi",
    default=None,
    type=str,
    help=(
        "Name of reference region. "
        "Required when PET is specified as a tsv file. "
        "REFROI option cannot be used if PET is specified as an image."
    ),
)
@click.option(
    "--refmask",
    default=None,
    type=str,
    help=(
        "3-D binary mask indicating reference tissue (must be in alignment with PET). "
        "REFMASK option cannot be used if PET is specified as a tsv file."
    ),
)
@click.option(
    "--outputdir",
    default=None,
    type=str,
    help=(
        "Directory in which to save each estimated parametric image. "
        "If not provided, they will be saved in the same directory as the PET image. "
        "Outputs will have a '_<model>_<parameter>' suffix."
    ),
)
@click.option(
    "--json",
    default=None,
    type=str,
    help=(
        "PET-BIDS json file. "
        "If not specified, it is assumed to have the same name as the PET "
        "image file, except with a .json extension."
    ),
)
@click.option(
    "--petmask",
    default=None,
    type=str,
    help=(
        "Binary mask specifying voxels where model should be fitted. "
        "Voxels outside this binary mask in the resulting parametric images "
        "will be set to NA."
    ),
)
@click.option(
    "--start", default=None, type=float, help="Start of time window for model"
)
@click.option("--end", default=None, type=float, help="End of time window for model")
@click.option(
    "--fwhm",
    default=None,
    type=float,
    help="Full width at half max in mm for smoothing, used for some models",
)
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
    model: str,
    refroi: str | None,
    refmask: str | None,
    outputdir: str | None,
    json: str | None,
    petmask: str | None,
    start: float | None,
    end: float | None,
    fwhm: float | None,
    weight_by: WEIGHT_OPTS = "frame_duration",
    integration_type: INTEGRATION_TYPE_OPTS = "trapz",
) -> None:
    """Fit a reference tissue model to a dynamic PET image or TACs.

    PET: 4-D PET image (can be 3-D if model is SUVR) or a 2-D tabular TACs tsv file
    """
    model = model.lower()

    froot, ext, addext = splitext_addext(pet)

    pet_img, reftac, petmask_img_mat = parse_kineticmodel_inputs(
        pet, json, refroi, refmask, petmask
    )

    if start is None:
        start = pet_img.start_time
    if end is None:
        end = pet_img.end_time
    if start != pet_img.start_time or end != pet_img.end_time:
        pet_img = pet_img.extract(start, end)
        reftac = reftac.extract(start, end)

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

    if outputdir is not None:
        bname = os.path.basename(froot)
        froot = os.path.join(outputdir, bname)

    if isinstance(pet_img, PETBIDSMatrix):
        output = froot + "_km-" + model.replace(".", "") + ext
        data = np.empty((len(km.parameters), pet_img.num_elements))
        for i, param in enumerate(km.parameters.keys()):
            data[i] = km.get_parameter(param)
        datat = data.T
        with open(output, "w") as f:
            tsvwriter = csv.writer(f, delimiter="\t")
            tsvwriter.writerow(["name"] + list(km.parameters.keys()))
            for i, elem in enumerate(pet_img.elem_names):
                tsvwriter.writerow([elem] + datat[i].tolist())
    else:
        # save estimated parameters as image
        for param in km.parameters.keys():
            res_img: SpatialImage = km.get_parameter(param)  # type: ignore
            output = (
                froot + "_km-" + model.replace(".", "") + "_kp-" + param + ext + addext
            )
            res_img.to_filename(output)

    # also need to save a json PET BIDS derivative file
    # TODO


def parse_kineticmodel_inputs(
    tac_object_file: str,
    tac_object_json: str | None = None,
    refroi: str | None = None,
    refmask: str | None = None,
    petmask: str | None = None,
) -> Tuple[PETBIDSImage | PETBIDSMatrix, TemporalMatrix, NumpyRealNumberArray | None]:
    """Parse kinetic model inputs.

    Args:
        tac_object_file: 3-D or 4-D PET image file or a 2-D tabular TACs tsv file
        tac_object_json: PET-BIDS json accompanying tac_object_file
                         (if not provided, will be assumed to have the same name
                          as tac_object_file, except with a .json extension)
        refroi: Used only if tac_object_file is a tsv file. Name of reference
                region as specified in the tsv file.
        refmask: Used only if tac_object_file is an image file. 3-D binary mask
                 indicating reference tissue (must be in alignment with PET).
        petmask: Used only if tac_object_file is an image file. Binary mask
                 specifying voxels where model should be fitted.

    Returns:
        tac_object: 3-D or 4-D PET image or 2-D TACs
        reftac: reference TAC
        petmask_img_mat: binary PET mask

    Raises:
        ValueError: refroi or refmask is not specified, or PET and mask are not
                    in the same space
    """
    if tac_object_file[-4:] == ".tsv":
        # if it is a tsv, then read as PETBIDSMatrix
        if not refroi:
            raise ValueError("refroi must be specified when tac_object is a tsv file")

        tac_object_tm: PETBIDSMatrix = petbidsmatrix_load(
            tac_object_file, tac_object_json
        )
        reftac = tac_object_tm.get_elem(refroi)

        return tac_object_tm, reftac, None

    # otherwise, read as PETBIDSImage
    if not refmask:
        raise ValueError("refmask must be specified when tac_object is an image")

    tac_object: PETBIDSImage = petbidsimage_load(tac_object_file, tac_object_json)
    refmask_img: SpatialImage = nib_load(refmask)  # type: ignore
    # check that refmask is in the same space as pet
    if not np.all(tac_object.img.affine == refmask_img.affine):
        raise ValueError("PET and refmask are not in the same space")
    refmask_img_mat: NumpyRealNumberArray = refmask_img.get_fdata().astype("bool")

    petmask_img_mat: NumpyRealNumberArray | None
    if petmask:
        petmask_img: SpatialImage = nib_load(petmask)  # type: ignore
        # check that petmask is in the same space as pet
        if not np.all(tac_object.img.affine == petmask_img.affine):
            raise ValueError("PET image and petmask are not in the same space")

        petmask_img_mat = petmask_img.get_fdata().astype("bool")
        # check that refmask is fully within petmask
        if not np.all(petmask_img_mat[refmask_img_mat]):
            warnings.warn(
                (
                    "refmask is not fully within petmask. "
                    "Will restrict refmask to within petmask."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            refmask_img_mat = refmask_img_mat * petmask_img_mat
    else:
        petmask_img_mat = None

    # extract average TAC in reference region
    reftac = tac_object.mean_timeseries_in_mask(refmask_img_mat)

    return tac_object, reftac, petmask_img_mat


if __name__ == "__main__":
    kineticmodel(prog_name="kineticmodel")  # pragma: no cover
