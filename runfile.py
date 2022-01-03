import os
from sample_solutions import load_distribution, sample_solutions, \
    get_parameter_values
from slurf.scenario_problem import compute_solution_sets
from slurf.commons import path, getTime
from slurf.parser import parse_arguments
from slurf.plot import plot_results

if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Interpret arguments provided
    args = parse_arguments()
    
    print("\n===== Script started at:", getTime())
    
    # Load probability distribution data
    model_file, param_dic, properties, prop_labels, reliability, Tlist = \
        load_distribution(root_dir, args.model)
    
    # Sample parameter values
    param_values = get_parameter_values(args.Nsamples, param_dic)
        
    # Compute solutions by verifying the instantiated CTMCs
    sampler, solutions = sample_solutions(Nsamples = args.Nsamples, 
                                          model = path(root_dir, "models", model_file),
                                          properties = properties,
                                          param_list = list(param_dic.keys()),
                                          param_values = param_values,
                                          root_dir = root_dir,
                                          cache = False)
    
    print("\n===== Sampling completed at:", getTime())
    
    # Compute solution set using scenario optimization
    regions = compute_solution_sets(solutions, 
                                    beta = args.beta, 
                                    rho_min = args.rho_min, 
                                    increment_factor = args.rho_incr,
                                    itermax = args.rho_max_iter)
    
    print("\n===== Scenario problems completed at:", getTime())
    
    plot_results(root_dir, args, regions, solutions, reliability, prop_labels, 
                 Tlist)
    
    print("\n===== Plotting completed at:", getTime())