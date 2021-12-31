import numpy as np
import os

from sample_solutions import sample_solutions
from slurf.scenario_problem import compute_slurf, plot_slurf
from slurf.commons import getTime

root_dir = os.path.dirname(os.path.abspath(__file__))

def _path(folder, file):
    """
    Internal method for simpler listing of examples.
    :param folder: Folder.
    :param file: Example file.
    :return: Complete path to example file.
    """
    
    return os.path.join(root_dir, folder, file)

seed = False
if seed:
    np.random.seed(15)

print("Script started at:", getTime())

# Specify properties
Tlist = np.arange(5, 140+1, 5)

modelfile = "sir20.sm"
properties = ("done", Tlist)
    
# Compute solutions
sampler, solutions = sample_solutions(Nsamples = 200, 
                                      model = _path("models", modelfile),
                                      properties = properties,
                                      root_dir = root_dir,
                                      cache = False)

print("Sampling completed at:", getTime())

# Compute solution set using scenario optimization
regions = compute_slurf(Tlist, solutions, 
                        beta = 0.99, 
                        rho_min = 0.0001, 
                        increment_factor = 1.5,
                        itermax = 20)

# Plot the solution set
plot_slurf(Tlist, regions, solutions, mode='smooth')

print("Script done at:", getTime())
