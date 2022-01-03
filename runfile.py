import numpy as np
import os

from sample_solutions import sample_solutions, get_parameter_values
from slurf.scenario_problem import compute_solution_sets, plot_reliability
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

# Specify number of samples
Nsamples = 100

##########################

# WIP preset function to choose a model
preset = 3

if preset == 1:
    modelfile = "sir20.sm"
    properties = ("done", Tlist)
    param_list = ['ki', 'kr']
    
    param_dic = {
        'ki': {'type': 'interval', 'lower_bound': 0.05, 'upper_bound': 0.08},
        'kr': {'type': 'gaussian', 'mean': 0.065, 'covariance': 0.01, 'nonzero': True}
        }
    
    param_values = get_parameter_values(Nsamples, param_dic)
    
elif preset == 2:
    modelfile = "walkers_ringLL.sm"
    properties = ['R{"steps"}=? [ C<=T ]', 'P=? [F[T,T] (w1=17) ]']
    param_list = ['failureRate']
    
    param_values = np.full((Nsamples, 1), 0.3)
    
elif preset == 3:
    modelfile = "tandem.sm"
    
    properties = ['R=?[S]']
    param_list = ['mu1a', 'mu1b', 'mu2', 'kappa']
    
    param_dic = {
        'mu1a': {'type': 'interval', 'lower_bound': 0.1*2-0.1, 'upper_bound': 0.1*2+0.1},
        'mu1b': {'type': 'interval', 'lower_bound': 0.9*2-0.3, 'upper_bound': 0.9*2+0.3},
        'mu2': {'type': 'interval', 'lower_bound': 2-0.5, 'upper_bound': 2+0.5},
        'kappa': {'type': 'interval', 'lower_bound': 4-1, 'upper_bound': 4+1}
        }
    
    param_values = get_parameter_values(Nsamples, param_dic)
    
elif preset == 4:
    modelfile = "kanban.sm"
    
    properties = ['R{"tokens_cell1"}=? [ S ]', 'R{"tokens_cell2"}=? [ S ]']
    param_list = ['t']
    
    param_values = np.full((Nsamples, 1), 2)
    
##########################
    
# Compute solutions
sampler, solutions = sample_solutions(Nsamples = Nsamples, 
                                      model = _path("models", modelfile),
                                      properties = properties,
                                      param_list = param_list,
                                      param_values = param_values,
                                      root_dir = root_dir,
                                      cache = False)

print("Sampling completed at:", getTime())

# Compute solution set using scenario optimization
regions = compute_solution_sets(Tlist, solutions, 
                                beta = 0.99, 
                                rho_min = 0.0001, 
                                increment_factor = 1.5,
                                itermax = 20)

# Plot the solution set
plot_reliability(Tlist, regions, solutions, mode='smooth')

print("Script done at:", getTime())
