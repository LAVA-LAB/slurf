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

`python runfile.py`




