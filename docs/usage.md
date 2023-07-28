# Usage

_Dynamic PET_ provides two CLI functions: `denoise` and `kineticmodel`.

`denoise` operates on 4-D images only.
Currently, only the [HYPR-LR] algorithm is available.

`kineticmodel` is a flexible function that can operate on images or time activity curve (TAC) files.

## PET Inputs

### Images

PET image inputs to _Dynamic PET_ functions should be 4-D images with the last dimension corresponding to time.
The only exception to this is standardized uptake value ratio (SUVR) calculation (`kineticmodel --model SUVR`), which can operate on 3-D images (i.e., without a time dimension) as well as on 4-D images.

### Time activity curves (TACs)

Time activity curves should be stored in a tab-separated values (tsv) file where
columns correspond to regions (or voxels) and rows correspond to time frames.
Column headers should indicate the region (or voxel) name.
There should not be any row headers.
Time frame information (i.e., frame start, frame duration) should not be included
in this file, as these will be extracted from the accompanying `.json` file (see below).

```{note} **Example TAC file content**
| ROI1 | MyFavBrainRegion |
| --- | --- |
| 2.29 | 2.73 |
| 14.8 | 9.73 |
| 23.5 | 17.4 |
...
```

### Time framing information

Both PET image and TAC inputs should be accompanied by a `.json` file that follows the [PET-BIDS] specification. The `.json` file is not checked for compliance with the PET-BIDS specification, but should have the following fields for full functionality with _Dynamic PET_:

```{code-block} python
TracerRadionuclide: str

ScanStart: float
InjectionStart: float
FrameTimesStart: List[float]
FrameDuration: List[float]
```

## Execution

```{eval-rst}
.. click:: dynamicpet.__main__:denoise
   :prog: denoise
   :nested: full
```

```{eval-rst}
.. click:: dynamicpet.__main__:kineticmodel
    :prog: kineticmodel
    :nested: full
```

[hypr-lr]: https://doi.org/10.2967/jnumed.109.073999
[pet-bids]: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/09-positron-emission-tomography.html
