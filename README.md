# SLURF
### Scenario optimization for Learning Uncertain Reliability Functions


## Installation

- Install [Storm](https://www.stormchecker.org/documentation/obtain-storm/build.html) and [Stormpy](https://moves-rwth.github.io/stormpy/installation.html#installation-steps) using the instructions in the documentation.
  Note that one must use the master branches of both tools.
  Preferably, install these in a virtual environment.

- Install dependencies: on macOS, tkinter needs to be available.
  It can be installed via [Homebrew](https://brew.sh/):

  `brew install python-tk`

- Install Slurf using

  `python setup.py install`

- Run the tests using

  `pytest test`

## Usage

SLURF can be run with

`python runfile.py --N=<number of samples> --beta=<confidence probability> --folder=<subfolder to load model from> --model=<filename of model>`

For example, to run for `N=100` samples of the SIR epidemic model with a population of 20 and a confidence probability of `beta=0.99` (i.e., the obtained results via scenario optimization are correct with at least 99% probability), the following command may be executed:

`python runfile.py --N=100 --beta=0.99 --folder=epidemic --model=sir20.sm`

The `folder` and `model` arguments are mandatory, while the number of samples is `N=100` by default, and the default confidence probability if `beta=0.99`.

## Inspecting results

All results (including figures) are stored in the `output/` folder.
