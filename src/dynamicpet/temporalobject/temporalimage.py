"""TemporalImage class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from nibabel.funcs import concat_images
from nibabel.imageclasses import spatial_axes_first

from .temporalmatrix import TemporalMatrix
from .temporalobject import (
    INTEGRATION_TYPE_OPTS,
    WEIGHT_OPTS,
    TemporalObject,
    TimingError,
    check_frametiming,
)

if TYPE_CHECKING:
    from nibabel.spatialimages import SpatialImage
    from numpy.typing import NDArray

    from dynamicpet.typing_utils import NumpyNumberArray, RealNumber


class TemporalImage(TemporalObject["TemporalImage"]):
    """4-D image with corresponding time frame information.

    Args:
        img: a SpatialImage object with a 3-D or 4-D dataobj
        frame_start: vector containing the start time of each frame
        frame_duration: vector containing the duration of each frame

    Attributes:
        img: SpatialImage storing image data matrix and header
        frame_start: vector containing the start times of each frame
        frame_duration: vector containing durations of each frame

    """

    img: SpatialImage

    def __init__(
        self,
        img: SpatialImage,
        frame_start: NumpyNumberArray,
        frame_duration: NumpyNumberArray,
    ) -> None:
        """4-D image with corresponding time frame information.

        Args:
            img: a SpatialImage object with a 3-D or 4-D dataobj
            frame_start: vector containing the start time of each frame
            frame_duration: vector containing the duration of each frame

        Raises:
            ValueError: Image is not 3-D or 4-D with spatial axes first
            TimingError: Image has inconsistent timing info

        """
        if not spatial_axes_first(img):
            msg = (
                "Cannot create TemporalImage from SpatialImage with "
                "unknown spatial axes"
            )
            raise ValueError(msg)

        check_frametiming(frame_start, frame_duration)

        self.frame_start: NDArray[np.double] = np.array(frame_start, dtype=np.double)
        self.frame_duration: NDArray[np.double] = np.array(
            frame_duration,
            dtype=np.double,
        )

        self.img: SpatialImage
        if img.ndim == 3:  # noqa: PLR2004
            # if image is 3-D, store data matrix with a single element in 4th dim
            self.img = img.slicer[..., np.newaxis]
        elif img.ndim == 4:  # noqa: PLR2004
            self.img = img
        else:
            msg = "Image must be 3-D or 4-D"
            raise ValueError(msg)

        if self.img.shape[3] != len(self.frame_start):
            msg = (
                f"4th dimension of image ({self.img.shape[3]}) must match "
                f"the number of columns ({len(self.frame_start)}) in "
                "frame timing information"
            )
            raise TimingError(msg)

    @property
    def dataobj(self) -> NDArray[np.double]:
        """Get dataobj of image."""
        return self.img.get_fdata()

    @property
    def num_voxels(self) -> int:
        """Get number of voxels in each frame."""
        return self.num_elements

    def extract(self, start_time: RealNumber, end_time: RealNumber) -> TemporalImage:
        """Extract a temporally shorter TemporalImage from a TemporalImage.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, inclusive

        Returns:
            extracted TemporalImage

        """
        start_index, end_index = self.get_idx_extract_time(start_time, end_time)

        extracted_img: SpatialImage = self.img.slicer[:, :, :, start_index:end_index]

        return TemporalImage(
            extracted_img,
            self.frame_start[start_index:end_index],
            self.frame_duration[start_index:end_index],
        )

    def dynamic_mean(
        self,
        weight_by: WEIGHT_OPTS | NumpyNumberArray | None = None,
        integration_type: INTEGRATION_TYPE_OPTS = "rect",
    ) -> SpatialImage:
        """Compute the (weighted) dynamic mean over time.

        Args:
            weight_by: If weight_by == None, each frame is weighted equally.
                       If weight_by == 'frame_duration', each frame is weighted
                       proportionally to its duration (inverse variance weighting).
                       If weight_by is a 1-D array, then specified values are used.
            integration_type: rect (rectangular) or trapz (trapezoidal).

        Returns:
            3-D image that is the weighted temporal average

        """
        dyn_mean = self._dynamic_mean(weight_by, integration_type)

        # mean image
        return image_maker(dyn_mean, self.img)

    def concatenate(self, other: TemporalImage) -> TemporalImage:
        """Concatenate another TemporalImage at the end (in time).

        Args:
            other: TemporalImage to concatenate

        Returns:
            concatenated temporal image

        Raises:
            TimingError: TemporalImages have temporal overlap or
                         TemporalImage being concatenated is earlier in time

        """
        if self.overlap_with(other) != []:
            msg = "Cannot concatenate TemporalImages with temporal overlap"
            raise TimingError(msg)
        if self.end_time >= other.start_time:
            msg = "TemporalImage being concatenated occurs earlier in time"
            raise TimingError(msg)

        concat_img: SpatialImage = concat_images([self.img, other.img], axis=-1)  # type: ignore[no-untyped-call]
        return TemporalImage(
            concat_img,
            np.concatenate([self.frame_start, other.frame_start]),
            np.concatenate([self.frame_duration, other.frame_duration]),
        )

    def timeseries_in_mask(
        self,
        mask: NumpyNumberArray | None = None,
    ) -> TemporalMatrix:
        """Get time activity curves (TAC) for each voxel within a region of interest.

        Args:
            mask: 3-D binary mask

        Returns:
            timeseries (in mask if provided, otherwise in entire image)

        Raises:
            ValueError: binary mask is incompatible

        """
        # stack voxelwise TACs as rows of a 2-D matrix
        if mask is None:
            dataobj = self.dataobj.reshape((self.num_elements, self.num_frames))
        elif mask.shape == self.dataobj.shape[:-1]:
            dataobj = self.dataobj[mask.astype("bool"), :]
        else:
            msg = "Binary mask is incompatible with data"
            raise ValueError(msg)

        # TACs
        return TemporalMatrix(
            dataobj,
            self.frame_start,
            self.frame_duration,
        )

    def mean_timeseries_in_mask(self, mask: NumpyNumberArray) -> TemporalMatrix:
        """Get mean time activity curve (TAC) within a region of interest.

        Args:
            mask: 3-D binary mask

        Returns:
            mean time series in mask

        """
        return TemporalMatrix(
            self.dataobj[mask.astype("bool"), :].mean(axis=0),
            self.frame_start,
            self.frame_duration,
        )


def image_maker(x: NumpyNumberArray, img: SpatialImage) -> SpatialImage:
    """Make image from dataobj.

    Args:
        x: data object
        img: image whose class, affine, and header will be used to
                make x into an image

    Returns:
        created image

    """
    return img.__class__(x, img.affine, img.header)
