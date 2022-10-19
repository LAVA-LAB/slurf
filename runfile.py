#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a Python implementation of the approach proposed in the paper:
    
 "Sampling-Based Verification of CTMCs with Uncertain Rates"
 
written by Thom Badings, Nils Jansen, Sebastian Junges, Marielle Stoelinga,
and Matthias Volk.
 
Contact e-mail address:     thom.badings@ru.nl
______________________________________________________________________________
"""

import os
import pandas as pd
import time
import numpy as np

from slurf.solution_sampler import load_distribution, sample_solutions, \
    get_parameter_values, validate_solutions, refine_solutions
from slurf.load_valuations import load_fixed_valuations
from slurf.scenario_problem import compute_confidence_region, \
    compute_confidence_per_dim, refinement_scheme, init_rho_list
from slurf.markov_chain_sampler import MarkovChainSamplerInterface
from slurf.dft_sampler import DftParametricModelSamplerInterface, \
    DftConcreteApproximationSamplerInterface, DftSimulationSamplerInterface
from slurf.approximate_checker import ApproxHeuristic
from slurf.util import path, getTime, print_stats, set_solution_df, \
    set_output_path
from slurf.parser import parse_arguments
from slurf.export import plot_results, save_results, export_benchmark_table

if __name__ == '__main__':

    print("\n===== SLURF started at:", getTime(), "=====")    

    # Set root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Parse arguments
    args = parse_arguments()
    
    # Set random seed according to the iteration
    np.random.seed(10+args.seed)

    # Initialize dictionary for storing all results
    dfs = {}
    
    # Define dictionary for exporting benchmark statistics
    expdata = {'name': args.modelfile_nosuffix, 'N': args.Nsamples}
    
    # Create output folder 
    output_path = set_output_path(root_dir, args)
    
    # Load probability distribution data
    model_file, param_dic, properties, args.prop_labels, args.reliability, \
     args.timebounds = load_distribution(root_dir, args.model, args.model_type,
                              parameters_file=args.param_file, 
                              properties_file=args.prop_file)
    
    print("\n===== Script initialized at:", getTime(),"=====")
    
    # Sample parameter values from the probability distribution
    if 'values' not in list(param_dic.values())[0]:
        print('- Sample from distribution...')
        param_values = get_parameter_values(args.Nsamples, param_dic)
    else:
        print('- Load fixed parameter valuations...')
        param_values = load_fixed_valuations(param_dic, args.Nsamples)
        
        
    print(list(param_dic.keys()))
    print(param_values)
    
    print("\n===== Parameter values sampled at:", getTime(),"=====")
    
    # Sample parameter values
    if args.model_type == 'DTMC':
        sampler = MarkovChainSamplerInterface()
        expdata['instantiator'] = 'DTMC'
    elif args.model_type == 'CTMC':
        sampler = MarkovChainSamplerInterface()
        expdata['instantiator'] = 'CTMC'

    else:
        # Current hack to get cabinets to work
        all_relevant=True
        if args.dft_checker == 'parametric':
            # Build parametric model once and sample on (partial or 
            # complete) CTMC
            sampler = DftParametricModelSamplerInterface(all_relevant)  
            expdata['instantiator'] = 'DFT (parametric)'
        elif args.dft_checker == 'concrete':
            # Build partial models for each sample
            sampler = DftConcreteApproximationSamplerInterface(all_relevant)
            expdata['instantiator'] = 'DFT (concrete)'
        else:
            assert args.dft_checker == 'simulation'
            # Perform simulations for each sample
            sampler = DftSimulationSamplerInterface(all_relevant, no_simulation=1000)
            expdata['instantiator'] = 'DFT (simulation)'
            
    sampler.set_max_cluster_distance(1e-2)
    sampler.set_approximation_heuristic(ApproxHeuristic.REACH_PROB)

    print("\n===== Sampler initialized at:", getTime(),"=====")
    
    # Load model
    model_file = path(root_dir, "models", model_file)
    sampler.load(model_file, properties, bisim=args.bisim)
    
    # Compute solutions by verifying the instantiated CTMCs
    cache = str(args.modelfile_nosuffix) + '_N=' + str(args.Nsamples) + '_seed=' + str(args.seed) + '.pkl'
    sampleObj, solutions = sample_solutions( 
                            sampler = sampler,
                            Nsamples = args.Nsamples, 
                            properties = properties,
                            param_list = list(param_dic.keys()),
                            param_values = param_values,
                            root_dir = root_dir,
                            cache = cache, 
                            exact = args.exact)
    
    print(properties)
    print(solutions)
    
    # If approximate model checker is active, also refine solution vectors
    if not args.exact:
        toRefine = [r for r in np.arange(len(solutions)) 
                    if not sampleObj[r].is_refined()]
        solutions = refine_solutions(sampler, sampleObj, solutions, 
                                     toRefine, args.precision, dict())
    
    dfs['solutions'] = set_solution_df(args.exact, solutions)
    print("\n===== Sampler finished at:", getTime(),"=====")
    time_before_scenario = time.process_time()

    # Set the list of the costs of relaxations to be used
    rho_list = init_rho_list(args)
    
    if args.exact or not args.refine:
        expdata['model_checking_type'] = 'exact'
        
        # Compute solution set using scenario optimization
        regions, dfs['regions'], dfs['regions_stats'], _ = \
            compute_confidence_region(solutions, args.beta, args, rho_list)
            
    else:
        expdata['model_checking_type'] = 'approximate'
        
        # Enter iterative refinement scheme
        regions, dfs['regions'], dfs['regions_stats'], dfs['refinement'] =\
            refinement_scheme(output_path, sampler, sampleObj, solutions, 
                              args, rho_list, plotEvery=1, max_iter = 20)
        
    # Store time taken for solving scenario optimization problems
    expdata['scen_opt_time'] = np.round(
         time.process_time() - time_before_scenario, 2)
        
    if args.naive_baseline and args.exact:
        args.naive_bounds = compute_confidence_per_dim(solutions, args, 
                                                       [max(rho_list)])
        expdata['naive_bounds_max_rho'] = args.naive_bounds
        print('Naive bounds obtained by analyzing measures independently:', 
              args.naive_bounds)
    else:
        expdata['naive_bounds_max_rho'] = {}
    
    print("\n===== Scenario problems completed at:", getTime(),"=====")
    
    # Plot results
    plot_results(output_path, args, regions, solutions)
    
    print("\n===== Plotting completed at:", getTime(),"=====")
    
    # Validation of results
    if args.Nvalidate > 0:
        # Sample new parameter for validation
        param_values = get_parameter_values(args.Nvalidate, param_dic)
        
        emp_satprob = validate_solutions(sampler, regions, args.Nvalidate, 
               properties, list(param_dic.keys()), param_values,
               root_dir=root_dir, cache=args.modelfile_nosuffix+'_seed'+ \
                   str(args.seed)+'_cache.pkl')
                            
        dfs['regions_stats']['Emp_satprob'] = emp_satprob
                            
    else:
        emp_satprob = [0]*len(regions)
        
    print("\n===== Validation completed at:", getTime(),"=====")
    
    try:
        dfs['storm_stats'] = print_stats(sampler.get_stats())
    except:
        True
    
    # Save raw results in Excel file
    dfs['arguments'] = pd.Series(vars(args), name='value')
    save_results(output_path, dfs, args.modelfile_nosuffix, 
                 args.Nsamples, args.export_filetype)
        
    # Merge with the latest benchmark table csv export file
    if args.export_stats:
        export_benchmark_table(root_dir, args, dfs, expdata, regions, 
                               emp_satprob)
