import os
from sample_solutions import load_distribution, sample_solutions, \
    get_parameter_values
from slurf.scenario_problem import compute_solution_sets
from slurf.model_sampler_interface import \
    CtmcReliabilityModelSamplerInterface, DftReliabilityModelSamplerInterface
from slurf.commons import path, getTime
from slurf.parser import parse_arguments
from slurf.plot import plot_results

if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Interpret arguments provided
    # args = parse_arguments(manualModel="dft/hecs/hecs_2_1.dft")
    #args = parse_arguments(manualModel="dft/hemps/hemps.dft")
    #args = parse_arguments(manualModel="dft/rc/rc.1-1-hc.dft")
    args = parse_arguments()
    
    print("\n===== Script started at:", getTime(),"=====")
    
    # Load probability distribution data
    model_file, param_dic, properties, prop_labels, reliability, timebounds = \
        load_distribution(root_dir, args.model, args.model_type)
    
    print("\n===== Data loaded at:", getTime(),"=====")
    
    # Sample parameter values from the probability distribution
    param_values = get_parameter_values(args.Nsamples, param_dic)
    
    print("\n===== Parameter values sampled at:", getTime(),"=====")
    
    print(args.bisim)
    
    # Sample parameter values
    if args.model_type == 'CTMC':
        sampler = CtmcReliabilityModelSamplerInterface()
    else:
        sampler = DftReliabilityModelSamplerInterface()
    
    print("\n===== Sampler initialized at:", getTime(),"=====")
    
    # Compute solutions by verifying the instantiated CTMCs
    sampler, solutions = sample_solutions( sampler = sampler,
                            Nsamples = args.Nsamples, 
                            model = path(root_dir, "models", model_file),
                            bisim = args.bisim,
                            properties = properties,
                            param_list = list(param_dic.keys()),
                            param_values = param_values,
                            root_dir = root_dir,
                            cache = False )
    
    print("\n===== Sampling from model completed at:", getTime(),"=====")
    
    # Compute solution set using scenario optimization
    regions = compute_solution_sets(solutions, 
                                    beta = args.beta, 
                                    rho_min = args.rho_min, 
                                    increment_factor = args.rho_incr,
                                    itermax = args.rho_max_iter)
    
    print("\n===== Scenario problems completed at:", getTime(),"=====")
    
    plot_results(root_dir, args, regions, solutions, reliability, prop_labels, 
                 timebounds)
    
    print("\n===== Plotting completed at:", getTime(),"=====")