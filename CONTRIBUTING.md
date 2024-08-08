# Contributor Guide

Thank you for your interest in improving this project.
This project is open-source under the [MIT license] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code]
- [Documentation]
- [Issue Tracker]
- [Code of Conduct]

[mit license]: https://opensource.org/licenses/MIT
[source code]: https://github.com/bilgelm/dynamicpet
[documentation]: https://dynamicpet.readthedocs.io/
[issue tracker]: https://github.com/bilgelm/dynamicpet/issues

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## How to set up your development environment

You need Python 3.11+ and the following tools:

- [Poetry]
- [Nox]
- [nox-poetry]

Install the package with development requirements:

```console
$ poetry install --all-extras
```

You can now run an interactive Python session,
or the command-line interface:

```console
$ poetry run denoise --help
$ poetry run kineticmodel --help
```

[poetry]: https://python-poetry.org/
[nox]: https://nox.thea.codes/
[nox-poetry]: https://nox-poetry.readthedocs.io/

## How to test the project

Run the full test suite:

```console
$ nox
```

List the available Nox sessions:

```console
$ nox --list-sessions
```

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

```console
$ nox --session=tests
```

The full test suite currently runs tests in Python 3.11 and 3.12.
You can also run a specific Python version only like this:

```console
$ nox -p=3.12
```

Unit tests are located in the _tests_ directory,
and are written using the [pytest] testing framework.

[pytest]: https://pytest.readthedocs.io/

## How to submit changes

Open a [pull request] to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation accordingly.

Feel free to submit early, though—we can always iterate on this.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command:

```console
$ nox --session=pre-commit -- install
```

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

[pull request]: https://github.com/bilgelm/dynamicpet/pulls

## Contributing a new kinetic model implementation to the toolbox

The `.py` file implementing your kinetic model should be located under the
`src/dynamicpet/kineticmodel/` directory.
`kineticmodel.py` in this directory implements the abstract base class that all
kinetic model implementations should inherit from:

```python
from .kineticmodel import KineticModel

class MyKineticModelImplementation(KineticModel):
    ...
```

Your kinetic model class needs to implement the following functions:

1. `get_param_names`

   This is a class method that returns a list of the names of kinetic model
   parameters.

   ```python
   @classmethod
   get_param_names(cls) -> list[str]:
       return ["parameter1", "parameter2"]
   ```

   In the `fit` function below, all model parameters generated
   and stored in the kinetic model class should be listed here.

2. `fit`

   This method performs kinetic model fitting (within `mask`, if provided)
   and stores estimated parameters using the`set_parameter` function.

   `mask` is a
   1-D (for `TemporalMatrix` time activity curves, or TACs) or
   3-D (for `TemporalImage` TACs)
   `NumpyRealNumberArray` and should match the dimensions of the
   `TemporalMatrix` or `TemporalImage` provided for the `tacs` attribute of
   the kinetic model. For zero values in `mask`, the time activity curve of the
   corresponding element of `tac` will be ignored when fitting the kinetic
   model. When `mask = None`, all `tac` elements will be used.

   This method does not return anything.

   ```python
   from ..temporalobject.temporalmatrix import TemporalMatrix
   from ..typing_utils import NumpyRealNumberArray

   def fit(self, mask: NumpyRealNumberArray | None = None) -> None:
       # perform kinetic model fitting here
       tacs: TemporalMatrix = self.tacs.timeseries_in_mask(mask)

       # in an actual implementation, you would estimate parameters
       # in this toy example, we set param1 and param2 to all 1's and 2's
       # (tacs.num_elements is the number of regions or voxels in tacs)
       param1 = np.ones((tacs.num_elements, 1))
       param2 = 2 * np.ones((tacs.num_elements, 1))

       # save the parameters
       self.set_parameter("parameter1", param1, mask)
       self.set_parameter("parameter2", param2, mask)
   ```

3. `fitted_tacs`

   This method computes fitted time activity curves using the estimated
   parameters. The output of this method should match the dimensions and type
   of TACs supplied (`TemporalImage` or `TemporalMatrix`).

   This method is optional.

   ```python
   from ..temporalobject.temporalimage import TemporalImage
   from ..temporalobject.temporalimage import image_maker

   def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
       """Get fitted TACs based on estimated model parameters."""
       # in an actual implementation, you would calculate fitted values from
       # estimated parameters
       # in this toy example, we just set it to an empty array
       fitted_tacs_dataobj = np.empty_like(self.tacs.dataobj)

       if isinstance(self.tacs, TemporalImage):
           img = image_maker(fitted_tacs_dataobj, self.tacs.img)
           ti = TemporalImage(img, self.tacs.frame_start, self.tacs.frame_duration)
           return ti
       else:
           tm = TemporalMatrix(
               fitted_tacs_dataobj, self.tacs.frame_start, self.tacs.frame_duration
           )
           return tm
   ```

You will also need to write tests with sufficient coverage to verify that your
implementation runs correctly.
For some examples, see `tests/test_kineticmodel.py`.

Finally, you'll need to edit the `kineticmodel` function in `__main__.py` so
that your model implementation can be executed using the command line interface.

<!-- github-only -->

[code of conduct]: CODE_OF_CONDUCT.md
