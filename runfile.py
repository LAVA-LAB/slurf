import numpy as np
import os

from sample_solutions import sample_solutions, get_parameter_values
from slurf.scenario_problem import compute_solution_sets
from slurf.plot import plot_reliability, plot_solution_set_2d
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


# Specify number of samples
Nsamples = 250

# Specify confidence level
beta = 0.99

##########################

# WIP preset function to choose a model
preset = 2

if preset == 1:
    modelfile = "sir20.sm"
    
    Tlist = np.arange(5, 140+1, 5)
    properties = ("done", Tlist)
    
    param_list = ['ki', 'kr']
    param_dic = {
        'ki': {'type': 'interval', 'lb': 0.05, 'ub': 0.08},
        'kr': {'type': 'gaussian', 'mean': 0.065, 'std': 0.01, 'nonzero': True}
        }
    
    param_values = get_parameter_values(Nsamples, param_dic)
    
elif preset == 2:
    modelfile = "tandem7.sm"
    
    t = 3000
    properties = ['R=? [ I='+str(t)+' ]', 
                  'P=? [ F<='+str(t)+' sc=c & sm=c & ph=2 ]']
    property_names = ['Exp. customers at time '+str(t),
                      'Prob. that networks becomes full in '+str(t)+' time units']
    param_list = ['lambdaF', 'mu1a', 'mu1b', 'mu2', 'kappa']
    
    var0 = 0.05
    var1 = 0.05
    var2 = 0.05
    var3 = 0.05
    var4 = 0.05
    
    param_dic = {
        'lambdaF': {'type': 'interval', 'lb': 4-var0, 'ub': 4+var0},
        'mu1a': {'type': 'interval', 'lb': 0.1*2-var1, 'ub': 0.1*2+var1},
        'mu1b': {'type': 'interval', 'lb': 0.9*2-var2, 'ub': 0.9*2+var2},
        'mu2': {'type': 'interval', 'lb': 2-var3, 'ub': 2+var3},
        'kappa': {'type': 'interval', 'lb': 4-var4, 'ub': 4+var4}
        }
    
    std0 = 0.1
    std1 = 0.1
    std2 = 0.1
    std3 = 0.1
    std4 = 0.1
    
    param_dic = {
        'lambdaF': {'type': 'gaussian', 'mean': 4, 'std': std0},
        'mu1a': {'type': 'gaussian', 'mean': 0.2, 'std': std1},
        'mu1b': {'type': 'gaussian', 'mean': 1.8, 'std': std2},
        'mu2': {'type': 'gaussian', 'mean': 2, 'std': std3},
        'kappa': {'type': 'gaussian', 'mean': 4, 'std': std4}
        }
    
    param_values = get_parameter_values(Nsamples, param_dic)
    
    plot_idxs = (0,1)
    
elif preset == 3:
    modelfile = "kanban.sm"
    
    properties = ['R{"tokens_cell1"}=? [ S ]', 'R{"tokens_cell2"}=? [ S ]']
    param_list = ['t']
    param_values = np.full((Nsamples, 1), 2)
    
# elif preset == -1:
#     modelfile = "walkers_ringLL.sm"
    
#     properties = ['R{"steps"}=? [ C<=10 ]', 'P=? [F[10,10] (w1=17) ]']
    
#     param_list = ['failureRate']
#     param_values = np.full((Nsamples, 1), 0.3)
    
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
regions = compute_solution_sets(solutions, 
                                beta = beta, 
                                rho_min = 0.0001, 
                                increment_factor = 1.5,
                                itermax = 20)

# Plot the solution set
if type(properties) == tuple:
    # As reliability over time (if properties object is a tuple)
    plot_reliability(Tlist, regions, solutions, beta, mode='smooth')
    
else:
    # As a solution set (if properties object is a list of properties)
    plot_solution_set_2d(plot_idxs, property_names, regions, solutions, 
                         beta, plotSamples=True)

print("Script done at:", getTime())
