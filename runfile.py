import numpy as np
import os

from sample_SIR import sample_SIR
from slurf.scenario_problem import compute_slurf
from slurf.commons import getTime

testfile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models")

def _path(folder, file):
    """
    Internal method for simpler listing of examples.
    :param folder: Folder.
    :param file: Example file.
    :return: Complete path to example file.
    """
    
    return os.path.join(testfile_dir, folder, file)


np.random.seed(15)

start, startTime = getTime()
print("Script started at:", startTime)

# Generate samples
Tlist = np.arange(10, 140+1, 5)

Nsamples = 100

# Generate given number of solutions to the parametric model
model = _path("", "sir20.sm")

sampler, sampleIDs, results = sample_SIR(Nsamples, Tlist, model)

min_rho = 0.004
increment_factor = 2
beta = 0.9

# Compute SLURF and plot
Pviolation, x_low, x_upp = compute_slurf(Tlist, results, beta,
                                         min_rho, increment_factor)

end, endTime = getTime()
print("Script done at:", endTime)
