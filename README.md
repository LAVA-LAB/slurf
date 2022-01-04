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

`python runfile.py --N=<number of samples> --beta=<confidence probability> --model=<path to model file>`

The `folder` argument should contain the path to the model file (e.g., in PRISM format), rooted in the `model` folder. For example, to run for `N=100` samples of the SIR epidemic model with a population of 20 and a confidence probability of `beta=0.99` (i.e., the obtained results via scenario optimization are correct with at least 99% probability), the following command may be executed:

`python runfile.py --N=100 --beta=0.99 --folder=epidemic --model=epidemic/sir20.sm`

The `model` argument is mandatory, while the number of samples is `N=100` by default, and the default confidence probability if `beta=0.99`. Other optional arguments are as follows:

- `--rho_min 0.0001` - Minimum cost of violation to start with in first iteration (0.0001 by default)
- --`rho_incr` 1.5 - Increment factor of the cost of violation in every iteration (1.5 by default)
- --`rho_max_iter 20` - Maximum number of iterations for increasing cost of violations (20 by default)
- `--curve_plot_mode conservative` - Plotting mode for reliability curves. Is either `optimistic` or `conservative` (default)

For models with multiple reward structures, bisimulation should be disabled. You can do this by providing the optional argument `--no-bisim`.

## Inspecting results

All results (including figures) are stored in the `output/` folder.

