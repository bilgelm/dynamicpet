# Usage

PET image inputs to _Dynamic PET_ functions should be accompanied by a `.json` file that follows the [PET-BIDS] specification. The `.json` file is not checked for compliance with the PET-BIDS specification, but should have the following fields for full functionality:

```{eval-rst}
.. autoclass:: dynamicpet.petbids.petbidsjson.PetBidsJson
    :members:
    :undoc-members:
```

PET inputs should be 4-D images (with the last dimension corresponding to time), with the exception of PET input to standardized uptake value ratio (SUVR) calculation, which can be 3- or 4-D.

Reference tissue-based kinetic models require a binary mask that defines the reference region in PET space.

The CLI function for fitting kinetic models is `kineticmodel`:

```{eval-rst}
.. click:: dynamicpet.__main__:kineticmodel
    :prog: kineticmodel
    :nested: full
```

_Dynamic PET_ also provides denoising algorithms through its CLI function `denoise`:

```{eval-rst}
.. click:: dynamicpet.__main__:denoise
   :prog: denoise
   :nested: full
```

[pet-bids]: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/09-positron-emission-tomography.html
