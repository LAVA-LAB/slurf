import os
import pandas as pd
import time
import numpy as np
from slurf.sample_solutions import load_distribution, sample_solutions, \
    get_parameter_values, validate_solutions
from slurf.scenario_problem import compute_confidence_region
from slurf.model_sampler_interface import \
    CtmcReliabilityModelSamplerInterface, DftReliabilityModelSamplerInterface
from slurf.commons import path, getTime, create_output_folder
from slurf.parser import parse_arguments
from slurf.export import plot_results, save_results

if __name__ == '__main__':

    np.random.seed(10)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    dfs = {}
    timing = {}
    
    # Interpret arguments provided
    args = parse_arguments(manualModel='ctmc/buffer/buffer.sm')
    # args = parse_arguments()
    
    print("\n===== Script started at:", getTime(),"=====")
    time_start = time.process_time()
    
    # Load probability distribution data
    model_file, param_dic, properties, prop_labels, reliability, timebounds = \
        load_distribution(root_dir, args.model, args.model_type,
                          parameters_file=args.param_file, 
                          properties_file=args.prop_file)
    
    timing['1_load'] = time.process_time() - time_start
    print("\n===== Data loaded at:", getTime(),"=====")
    time_start = time.process_time()
    
    # Sample parameter values from the probability distribution
    param_values = get_parameter_values(args.Nsamples, param_dic)
    
    timing['2_param_sampling'] = time.process_time() - time_start
    print("\n===== Parameter values sampled at:", getTime(),"=====")
    time_start = time.process_time()
    
    # Sample parameter values
    if args.model_type == 'CTMC':
        sampler = CtmcReliabilityModelSamplerInterface()
    else:
        sampler = DftReliabilityModelSamplerInterface()
    
    timing['3_init_sampler'] = time.process_time() - time_start
    print("\n===== Sampler initialized at:", getTime(),"=====")
    time_start = time.process_time()
    
    # Load model
    file = path(root_dir, "models", model_file)
    sampler.load(file, properties, bisim=args.bisim)
    
    # Compute solutions by verifying the instantiated CTMCs
    solutions = sample_solutions( sampler = sampler,
                            Nsamples = args.Nsamples, 
                            properties = properties,
                            param_list = list(param_dic.keys()),
                            param_values = param_values,
                            root_dir = root_dir,
                            cache = False )
    
    dfs['solutions'] = pd.DataFrame(solutions)
    dfs['solutions'].index.names = ['Sample']
    
    timing['4_model_sampling'] = time.process_time() - time_start
    print("\n===== Sampler finished at:", getTime(),"=====")
    time_start = time.process_time()
    
    print('-----------------------------------------')
    print('MODEL STATISTICS:')
    stats = sampler.get_stats()
    dfs['storm_stats'] = pd.Series(stats)
    print(dfs['storm_stats'])
    print('-----------------------------------------')
    
    # %%
    
    args.pareto_pieces = 2
    
    # Compute solution set using scenario optimization
    regions, dfs['regions'], dfs['regions_stats'] = compute_confidence_region(
                                    solutions, args)
    
    timing['5_scenario_problems'] = time.process_time() - time_start
    print("\n===== Scenario problems completed at:", getTime(),"=====")
    time_start = time.process_time()
    
    # Create output folder
    output_folder = args.modelfile_nosuffix + \
                        "_N=" + str(args.Nsamples) + "_beta=" + str(args.beta)
    output_path = create_output_folder(root_dir, output_folder)
    
    # Plot results
    plot_results(output_path, args, regions, solutions, reliability, prop_labels, 
                 timebounds)
    
    timing['6_plotting'] = time.process_time() - time_start
    print("\n===== Plotting completed at:", getTime(),"=====")
    time_start = time.process_time()
    
    # Validation of results
    if args.Nvalidate > 0:

        print('- Validate confidence regions with',args.Nvalidate,'samples')        

        # Sample new parameter for validation
        param_values = get_parameter_values(args.Nvalidate, param_dic)
        
        Pemp = validate_solutions(sampler, regions, args.Nvalidate, properties, 
                           list(param_dic.keys()), param_values, root_dir,
                           cache = args.modelfile_nosuffix+'_cache.pkl')
        
        dfs['regions_stats']['Pviolation_empirical'] = Pemp
    
    timing['7_validation'] = time.process_time() - time_start
    print("\n===== Validation completed at:", getTime(),"=====")
    
    # Save raw results in Excel file
    dfs['timing'] = pd.Series(timing)
    save_results(output_path, dfs, args.modelfile_nosuffix, 
                 args.Nsamples, args.beta)