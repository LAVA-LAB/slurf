import os
import pandas as pd
import time
import numpy as np
import itertools
import copy

from slurf.sample_solutions import load_distribution, sample_solutions, \
    get_parameter_values, validate_solutions
from slurf.scenario_problem import compute_confidence_region
from slurf.model_sampler_interface import \
    CtmcReliabilityModelSamplerInterface, DftReliabilityModelSamplerInterface
from slurf.commons import path, getTime, print_stats, set_solution_df, \
    set_output_path, getDateTime
from slurf.parser import parse_arguments
from slurf.export import plot_results, save_results, Cases

if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.abspath(__file__))
    dfs = {}
    val_dfs = {}
    timing = {}
    
    # Interpret arguments provided
    # ARGS = parse_arguments(manualModel='ctmc/buffer/buffer.sm')
    ARGS = parse_arguments(manualModel='ctmc/epidemic/sir20.sm')
    
    # ARGS.Nsamples = [25,50,100]
    # ARGS.pareto_pieces = 9
    # ARGS.seeds = 2
    
    ARGS = parse_arguments()
    
    # Define dictionary over which to iterate
    iterate_dict = {'N': ARGS.Nsamples,
                    'seeds': np.arange(ARGS.seeds)}
    keys, values = zip(*iterate_dict.items())
    iterator = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    cases = Cases(root_dir)
    cases.init_eta('multi_iter_etas_'+str(getDateTime())+'.csv',
               ['#', 'N','seed','rho','empirical'] + list(map(str, ARGS.beta)))
    cases.init_time('multi_iter_ScenProbTime_'+str(getDateTime())+'.csv',
                    list(map(str, ARGS.Nsamples)))
    
    for q, itr in enumerate(iterator):
        
        args = copy.copy(ARGS)
        args.Nsamples = int(itr['N'])
        args.seeds = int(itr['seeds'])
        
        # Set random seed according to the iteration
        np.random.seed(itr['seeds'])
        
        print("\n===== Script started at:", getTime(),"=====")
        time_start = time.process_time()
        
        # Load probability distribution data
        model_file, param_dic, properties, prop_labels, reliability, timebounds = \
            load_distribution(root_dir, args.model, args.model_type,
                              parameters_file=args.param_file, 
                              properties_file=args.prop_file)
        
        timing['1_load'] = time.process_time() - time_start
        print("\n===== Data loaded at:", getTime(),"=====")
        
        # Create output folder 
        output_path = set_output_path(root_dir, args)
        
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
        
        dfs['solutions'] = set_solution_df(solutions)
        
        timing['4_model_sampling'] = time.process_time() - time_start
        print("\n===== Sampler finished at:", getTime(),"=====")
        time_start = time.process_time()
        
        dfs['storm_stats'] = print_stats(sampler.get_stats())
        
        # Compute solution set using scenario optimization
        regions, dfs['regions'], dfs['regions_stats'] = compute_confidence_region(
                                        solutions, args)
        
        timing['5_scenario_problems'] = time.process_time() - time_start
        print("\n===== Scenario problems completed at:", getTime(),"=====")
        time_start = time.process_time()
        
        # Plot results
        plot_results(output_path, args, regions, solutions, reliability, prop_labels, 
                     timebounds)
        
        timing['6_plotting'] = time.process_time() - time_start
        print("\n===== Plotting completed at:", getTime(),"=====")
        time_start = time.process_time()
        
        # Validation of results
        if args.Nvalidate > 0:
            # Sample new parameter for validation
            param_values = get_parameter_values(args.Nvalidate, param_dic)
            
            emp_satprob, val_dfs = \
                validate_solutions(val_dfs, sampler, regions, args.Nvalidate, 
                   properties, list(param_dic.keys()), param_values)
                                #args.modelfile_nosuffix+'_cache.pkl')
                                
            dfs['regions_stats']['Emp_satprob'] = emp_satprob
                                
        else:
            emp_satprob = [None]*len(regions)
            
        timing['7_validation'] = time.process_time() - time_start
        print("\n===== Validation completed at:", getTime(),"=====")
        
        # Save raw results in Excel file
        dfs['timing'] = pd.Series(timing)
        save_results(output_path, dfs, args.modelfile_nosuffix, 
                     args.Nsamples)
    
        # Export validation results to csv (for easy plotting to paper)
        cases.add_eta_row(q, args, regions, emp_satprob)
        cases.add_time_row(args, timing['5_scenario_problems'])
    
        cases.write_time()
    
    ###
        