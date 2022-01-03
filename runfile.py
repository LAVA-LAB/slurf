import numpy as np
import os

import itertools
import pandas as pd
import matplotlib.pyplot as plt

from sample_solutions import sample_solutions, get_parameter_values
from slurf.scenario_problem import compute_solution_sets
from slurf.plot import plot_reliability, plot_solution_set_2d
from slurf.commons import getTime, getDateTime
from slurf.parser import parse_arguments

root_dir = os.path.dirname(os.path.abspath(__file__))

def _path(folder, file):
    """
    Internal method for simpler listing of examples.
    :param folder: Folder.
    :param file: Example file.
    :return: Complete path to example file.
    """
    
    return os.path.join(root_dir, folder, file)

def _load_distribution(model_folder, model_file):
    """
    Helper function to load probability distribution and parameter data
    :model_folder: Subfolder to load the model from
    :model_file: Filename of the model to load
    """
    
    distr_file = _path("models/"+str(model_folder), "parameters.xlsx")
    try:
        open(distr_file)
    except IOError:
        print("ERROR: Distribution file does not exist")
        
    model_file = _path("models/"+str(model_folder), model_file)
    try:
        open(model_file)
    except IOError:
        print("ERROR: Model file does not exist")
    
    # Read parameter sheet
    param_df = pd.read_excel(distr_file, sheet_name='Parameters', index_col=0)
    param_dic = param_df.to_dict('index')

    # Read property sheet
    property_df = pd.read_excel(distr_file, sheet_name='Properties')
    properties = property_df['property'].to_list()
    prop_labels = property_df['label'].to_list()
    
    if 'time' in property_df:
        reliability = True
        Tlist = property_df['time'].to_list()
    else:
        reliability = False
        Tlist = None
    
    return model_file, param_dic, properties, prop_labels, reliability, Tlist




if __name__ == '__main__':

    args = parse_arguments()
    
    print("Script started at:", getTime())
    
    # Load probability distribution data
    model_file, param_dic, properties, prop_labels, reliability, Tlist = \
        _load_distribution(args.folder, args.file)
    
    # Sample parameter values
    param_values = get_parameter_values(args.Nsamples, param_dic)
        
    # Compute solutions by verifying the instantiated CTMCs
    sampler, solutions = sample_solutions(Nsamples = args.Nsamples, 
                                          model = _path("models", model_file),
                                          properties = properties,
                                          param_list = list(param_dic.keys()),
                                          param_values = param_values,
                                          root_dir = root_dir,
                                          cache = False)
    
    print("Sampling completed at:", getTime())
    
    # Compute solution set using scenario optimization
    regions = compute_solution_sets(solutions, 
                                    beta = args.beta, 
                                    rho_min = 0.0001, 
                                    increment_factor = 1.5,
                                    itermax = 20)
    
    print("Scenario problems completed at:", getTime())
    
    # Plot the solution set
    if reliability:
        # As reliability over time (if properties object is a tuple)
        plot_reliability(Tlist, regions, solutions, args.beta, mode='smooth', 
                         plotSamples=True)
        
        # Save figure
        exp_file = args.folder+'_'+str(getDateTime()+'.pdf')
        filename = _path("output", exp_file)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(' - Reliability plot exported to:',exp_file)
        
    else:
        # As a solution set (if properties object is a list of properties)    
        for idx_pair in itertools.combinations(np.arange(len(prop_labels)), 2):
            # Plot the solution set for every combination of 2 properties
            
            plot_solution_set_2d(idx_pair, prop_labels, regions, solutions, 
                                 args.beta, plotSamples=True)
    
            # Save figure
            exp_file = args.folder+'_'+str(getDateTime()+'_'+str(idx_pair)+'.pdf')
            filename = _path("output", exp_file)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            print(' - 2D plot exported to:',exp_file)