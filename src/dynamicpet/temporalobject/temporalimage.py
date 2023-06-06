"""TemporalImage class."""

import numpy as np
from nibabel.funcs import concat_images
from nibabel.imageclasses import spatial_axes_first
from nibabel.spatialimages import SpatialImage
from numpy.typing import NDArray

from ..typing_utils import NumpyRealNumberArray
from ..typing_utils import RealNumber
from .temporalmatrix import TemporalMatrix
from .temporalobject import WEIGHT_OPTS
from .temporalobject import TemporalObject
from .temporalobject import TimingError
from .temporalobject import check_frametiming


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
        frame_start: NumpyRealNumberArray,
        frame_duration: NumpyRealNumberArray,
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
            raise ValueError(
                "Cannot create TemporalImage from SpatialImage with "
                "unknown spatial axes"
            )

        check_frametiming(frame_start, frame_duration)

        self.frame_start: NDArray[np.double] = np.array(frame_start, dtype=np.double)
        self.frame_duration: NDArray[np.double] = np.array(
            frame_duration, dtype=np.double
        )

        self.img: SpatialImage
        if img.ndim == 3:
            # if image is 3-D, store data matrix with a single element in 4th dim
            self.img = img.slicer[..., np.newaxis]
        elif img.ndim == 4:
            self.img = img
        else:
            raise ValueError("Image must be 3-D or 4-D")

        if not self.img.shape[3] == len(self.frame_start):
            raise TimingError(
                f"4th dimension of image ({self.img.shape[3]}) must match "
                f"the number of columns ({len(self.frame_start)}) in "
                "frame timing information"
            )

    @property
    def dataobj(self) -> NDArray[np.double]:
        """Get dataobj of image."""
        return np.array(self.img.dataobj, dtype=np.double)

    @property
    def num_voxels(self) -> int:
        """Get number of voxels in each frame."""
        return self.num_elements

    # def nontemporal_object_maker(self) -> Callable[[NumpyRealNumberArray],
    #                                                SpatialImage]:
    #     """Get a Callable for creating a dataobj for TemporalImage."""
    #     func: Callable[[NumpyRealNumberArray], SpatialImage] = \
    #         lambda data: self.img.__class__(
    #             data, self.img.affine, self.img.header)
    #     return func

    def extract(self, start_time: RealNumber, end_time: RealNumber) -> "TemporalImage":
        """Extract a temporally shorter TemporalImage from a TemporalImage.

        Args:
            start_time: time at which to begin, inclusive
            end_time: time at which to stop, inclusive

        Returns:
            extracted_img: extracted TemporalImage
        """
        start_index, end_index = self.get_idx_extract_time(start_time, end_time)

        extracted_img: SpatialImage = self.img.slicer[:, :, :, start_index:end_index]

        extract_res = TemporalImage(
            extracted_img,
            self.frame_start[start_index:end_index],
            self.frame_duration[start_index:end_index],
        )
        return extract_res

    def dynamic_mean(
        self,
        weight_by: WEIGHT_OPTS | NumpyRealNumberArray | None = None,
    ) -> SpatialImage:
        """Compute the (weighted) dynamic mean over time.

        Args:
            weight_by: If weight_by == None, each frame is weighted equally.
                       If weight_by == 'frame_duration', each frame is weighted
                       proportionally to its duration (inverse variance weighting).
                       If weight_by is a 1-D array, then specified values are used.

        Returns:
            3-D image that is the weighted temporal average
        """
        dyn_mean: NumpyRealNumberArray = np.average(
            self.dataobj, axis=-1, weights=self.get_weights(weight_by)
        )

        # Create a SpatialImage of the same class as self.img
        # image_maker = self.img.__class__
        # mean_img = image_maker(dyn_mean, self.img.affine, self.img.header)
        mean_img = image_maker(dyn_mean, self.img)

        return mean_img

    def concatenate(self, other: "TemporalImage") -> "TemporalImage":
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
            raise TimingError("Cannot concatenate TemporalImages with temporal overlap")
        if self.end_time >= other.start_time:
            raise TimingError("TemporalImage being concatenated occurs earlier in time")

        concat_img: SpatialImage = concat_images([self.img, other.img])  # type: ignore
        concat_res = TemporalImage(
            concat_img,
            np.concatenate([self.frame_start, other.frame_start]),
            np.concatenate([self.frame_duration, other.frame_duration]),
        )

        return concat_res

    def timeseries_in_mask(
        self, mask: NumpyRealNumberArray | None = None
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
            raise ValueError("Binary mask is incompatible with data")

        tacs = TemporalMatrix(
            dataobj,
            self.frame_start,
            self.frame_duration,
        )
        return tacs

    def mean_timeseries_in_mask(self, mask: NumpyRealNumberArray) -> TemporalMatrix:
        """Get mean time activity curve (TAC) within a region of interest.

        Args:
            mask: 3-D binary mask

        Returns:
            mean time series in mask
        """
        mean_tac = TemporalMatrix(
            self.dataobj[mask.astype("bool"), :].mean(axis=0),
            self.frame_start,
            self.frame_duration,
        )
        return mean_tac


def image_maker(x: NumpyRealNumberArray, img: SpatialImage) -> SpatialImage:
    """Make image from dataobj.

    Args:
        x: data object
        img: image whose class, affine, and header will be used to
                make x into an image

    Returns:
        created image
    """
    return img.__class__(x, img.affine, img.header)
