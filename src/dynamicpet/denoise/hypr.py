"""HYPR denoising."""
# from scipy.ndimage import uniform_filter
import numpy as np
from nibabel.processing import smooth_image
from nibabel.spatialimages import SpatialImage

from ..petbids.petbidsimage import PETBIDSImage


def hypr_lr(ti: PETBIDSImage, fwhm: float) -> PETBIDSImage:
    """HYPR-LR denoising for dynamic PET.

    HYPR-LR is short for HighlY constrained backPRojection for Local Reconstruction.

    Reference:
    Christian, B. T., Vandehey, N. T., Floberg, J. M., Mistretta, C. A. (2010).
    Dynamic PET denoising with HYPR processing. Journal of Nuclear Medicine, 51(7),
    1147-1154. https://doi.org/10.2967/jnumed.109.073999

    Args:
        ti: dynamic PET
        fwhm: full width at half max (in mm) of the Gaussian smoothing filter

    Returns:
        HYPR-LR denoised dynamic PET
    """
    # decay uncorrect the PET image
    i = ti.decay_uncorrect()

    # calculate duration-weighted frame average
    i_c: SpatialImage = i.dynamic_mean(weight_by="frame_duration")

    # convolve both ti and weighted average by a low-pass filter (3D boxcar)
    ixf: SpatialImage = smooth_image(i.img, fwhm)
    i_cxf: SpatialImage = smooth_image(i_c, fwhm)
    # ixf = uniform_filter(ti.img, size=(size, size, size, 1), mode='nearest')
    # i_cxf = uniform_filter(i_c.dataobj, size=size, mode='nearest')

    i_w_dataobj = ixf.dataobj / i_cxf.dataobj[..., np.newaxis]
    i_h_dataobj = i_w_dataobj * i_c.dataobj[..., np.newaxis]
    i_h: SpatialImage = i_c.__class__(i_h_dataobj, i_c.affine, i_c.header)

    ti_h = PETBIDSImage(i_h, i.json_dict)

    # decay correct the result
    return ti_h.decay_correct()
