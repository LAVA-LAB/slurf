import os
import numpy as np
from datetime import datetime

from sample_SIR import sample_SIR
from slurf.scenario_problem import compute_slurf


def _path(folder, file):
    """
    Internal method for simpler listing of examples.
    :param folder: Folder.
    :param file: Example file.
    :return: Complete path to example file.
    """
    testfile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models")
    return os.path.join(testfile_dir, folder, file)


def getTime():

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    return current_time


startTime = getTime()
print("Script started at:", startTime)

# Generate samples
Tlist = np.arange(10, 100+1, 5)

Nsamples = 300

# Generate given number of solutions to the parametric model
model = _path("", "sir.sm")
sampler, sampleIDs, results = sample_SIR(Nsamples, Tlist, model)

# Values of rho (cost of regret) at which to solve the scenario program
# f = 3
# rho_list = np.round([1, 1/(f**1), 1/(f**2), 1/(f**3), 1/(f**4)], 3)
rho_list = [1, 0.1, 0.06, 0.03, 0.015]

# Compute SLURF and plot
Pviolation, x_low, x_upp = compute_slurf(Tlist, results, rho_list, beta=0.99)

endTime = getTime()
print("Script done at:", endTime)
