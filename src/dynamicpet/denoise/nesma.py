"""NESMA denoising."""

import numpy as np
from tqdm import tqdm

from dynamicpet.petbids.petbidsimage import PETBIDSImage
from dynamicpet.temporalobject.temporalimage import image_maker
from dynamicpet.typing_utils import NumpyNumberArray


def nesma_semiadaptive(
    ti: PETBIDSImage,
    mask: NumpyNumberArray,
    window_half_size: tuple[int, int, int],
    thresh: float = 0.05,
) -> tuple[PETBIDSImage, NumpyNumberArray]:
    """NESMA denoising.

    NESMA is short for Nonlocal EStimation of multispectral MAgnitudes.

    Reference:
    Bouhrara M, Reiter DA, Maring MC, Bonny JM, Spencer RG. Use of the NESMA
    Filter to Improve Myelin Water Fraction Mapping with Brain MRI.
    J Neuroimaging. 2018;28(6):640-9. https://doi.org/10.1111/jon.12537

    Args:
        ti: 4-D image
        mask: 3-D boolean mask indicating where smoothing should be performed
        window_half_size: the size of search window centered around a voxel will
                          be 2 * window_half_size + 1 (any part of the window
                          that extends beyond the image will be truncated)
        thresh: a value between 0 and 1, optional (default = 0.05)

    Returns:
        s_nesma: filtered output image, 4-D
        vsm: voxel similarity map, 3-D indicates the number of voxels in the
             search window that were below the threshold

    Raises:
        ValueError: incorrect input(s)

    """
    if thresh < 0 or thresh > 1:
        msg = "thresh must be between 0 and 1"
        raise ValueError(msg)
    if ti.img.ndim != 4:  # noqa: PLR2004
        msg = "input image must be 4-D"
        raise ValueError(msg)
    if mask.ndim != 3:  # noqa: PLR2004
        msg = "mask must be 3-D"
        raise ValueError(msg)
    if not np.all(ti.img.shape[:-1] == mask.shape):
        msg = "input image and mask shapes must be compatible"
        raise ValueError(msg)

    m, n, o, _ = ti.img.shape

    if m <= window_half_size[0] or n <= window_half_size[1] or o <= window_half_size[2]:
        msg = "image size should be greater than window_half_size"
        raise ValueError(msg)

    s_nesma = np.copy(ti.dataobj)
    vsm = np.zeros(ti.shape[:-1])

    indices = np.where(mask)

    for i, j, k in tqdm(zip(*indices, strict=True), total=np.sum(mask)):  # type: ignore[call-overload]
        tmin = max(k - window_half_size[2], 0)
        tmax = min(k + window_half_size[2] + 1, o)

        rmin = max(i - window_half_size[0], 0)
        rmax = min(i + window_half_size[0] + 1, m)

        # index voxel signal vector
        x = ti.dataobj[i : i + 1, j : j + 1, k : k + 1, :]

        smin = max(j - window_half_size[1], 0)
        smax = min(j + window_half_size[1] + 1, n)

        # get the neighborhood around voxel
        nbhd = ti.dataobj[rmin:rmax, smin:smax, tmin:tmax, :]

        # Relative Manhattan distance calculation
        #
        # In the NESMA paper, the denominator includes the L1 norm of x only.
        # Here, we use the sum of the L1 norm of x and the L1 norm of the voxel
        # x is being compared to (+ a small number to prevent division by 0).
        #
        # This choice of denominator is motivated by the inequality that
        # |y - x| <= |x| + |y|
        # This way, the threshold is bounded above by 1.
        distance = np.sum(np.abs(nbhd - x), axis=-1) / (
            np.sum(np.abs(x)) + np.sum(np.abs(nbhd), axis=-1) + np.finfo(float).eps
        )

        pos = distance < thresh
        n_below_thresh = np.sum(pos)
        if n_below_thresh > 0:
            s_nesma[i, j, k, :] = np.sum(nbhd[pos, :], axis=0) / n_below_thresh
            vsm[i, j, k] = n_below_thresh

    return PETBIDSImage(image_maker(s_nesma, ti.img), ti.json_dict), vsm
