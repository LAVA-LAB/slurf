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
import itertools
import copy

from slurf.solution_sampler import load_distribution, sample_solutions, \
    get_parameter_values, validate_solutions, refine_solutions
from slurf.scenario_problem import compute_confidence_region, \
    compute_confidence_per_dim, refinement_scheme, init_rho_list
from slurf.ctmc_sampler import CtmcReliabilityModelSamplerInterface
from slurf.dft_sampler import DftParametricModelSamplerInterface, \
    DftConcreteApproximationSamplerInterface
from slurf.approximate_ctmc_checker import ApproxHeuristic
from slurf.commons import path, getTime, print_stats, set_solution_df, \
    set_output_path
from slurf.parser import parse_arguments
from slurf.export import plot_results, save_results, Cases, \
    export_benchmark_table

if __name__ == '__main__':

    # Set root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize dictionaries
    dfs = {}
    val_dfs = {}
    
    # Parse arguments
    ARGS = parse_arguments()
    
    # Define dictionary for variables for which we can pass multiple values.
    # We will iterate over these variables as iterations of the same experiment
    iterate_dict = {'N': ARGS.Nsamples,
                    'seeds': np.arange(ARGS.seeds)}
    keys, values = zip(*iterate_dict.items())
    iterator = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Define CSV files in which to store results
    cases = Cases(root_dir, ARGS)
    
    # Define DataFrame format for the main table
    STATS = pd.DataFrame(index=[ARGS.modelfile_nosuffix])
    STATS.index.name = 'instance'
    
    # Iterate over the 'iterator' variable
    for q, itr in enumerate(iterator):
        
        # Initialize the current iteration
        args = copy.copy(ARGS)
        args.Nsamples = int(itr['N'])
        args.seeds = int(itr['seeds'])
        
        # Set random seed according to the iteration
        np.random.seed(10+itr['seeds'])
        
        print("\n===== Iteration", q, "started at:", getTime(), "=====")
        
        # Load probability distribution data
        model_file, param_dic, properties, args.prop_labels, args.reliability,\
            args.timebounds = load_distribution(root_dir, args.model, 
                                  args.model_type,
                                  parameters_file=args.param_file, 
                                  properties_file=args.prop_file)
        
        print("\n===== Data loaded at:", getTime(),"=====")
        
        # Create output folder 
        output_path = set_output_path(root_dir, args)
        
        # Sample parameter values from the probability distribution
        param_values = get_parameter_values(args.Nsamples, param_dic)
        
        print("\n===== Parameter values sampled at:", getTime(),"=====")
        
        # Sample parameter values
        if args.model_type == 'CTMC':
            sampler = CtmcReliabilityModelSamplerInterface()
            STATS['instantiator'] = 'CTMC'
            
        else:
            if args.dft_checker == 'parametric':
                # Build parametric model once and sample on (partial or 
                # complete) CTMC
                sampler = DftParametricModelSamplerInterface()  
                STATS['instantiator'] = 'DFT (parametric)'
                
            else:
                # Build partial models for each sample
                sampler = DftConcreteApproximationSamplerInterface()
                STATS['instantiator'] = 'DFT (concrete)'
                
        sampler.set_max_cluster_distance(1e-2)
        sampler.set_approximation_heuristic(ApproxHeuristic.REACH_PROB)

        print("\n===== Sampler initialized at:", getTime(),"=====")
        
        # Load model
        file = path(root_dir, "models", model_file)
        sampler.load(file, properties, bisim=args.bisim)
        
        # Compute solutions by verifying the instantiated CTMCs
        sampleObj, solutions = sample_solutions( 
                                sampler = sampler,
                                Nsamples = args.Nsamples, 
                                properties = properties,
                                param_list = list(param_dic.keys()),
                                param_values = param_values,
                                root_dir = root_dir,
                                cache = False, 
                                exact = args.exact)
        
        if not args.exact:
            toRefine = [r for r in np.arange(len(solutions)) 
                        if not sampleObj[r].is_refined()]
            solutions = refine_solutions(sampler, sampleObj, solutions, 
                                         toRefine, args.precision, dict())
        
        dfs['solutions'] = set_solution_df(args.exact, solutions)
        print("\n===== Sampler finished at:", getTime(),"=====")
        time_before_scenario = time.process_time()

        rho_list = init_rho_list(args)
        
        if args.exact or not args.refine:
            STATS['model checking type'] = 'exact'
            
            # Compute solution set using scenario optimization
            regions, dfs['regions'], dfs['regions_stats'], _ = \
                compute_confidence_region(solutions, args.beta, args, rho_list)
                
        else:
            STATS['model checking type'] = 'approximate'
            
            # Enter iterative refinement scheme
            regions, dfs['regions'], dfs['regions_stats'], dfs['refinement'] =\
                refinement_scheme(output_path, sampler, sampleObj, solutions, 
                                  args, rho_list, plotEvery=1, max_iter = 20)
            
        # TODO make implementation of this baseline (experiment for paper) more refined
        if args.naive_baseline and args.exact:
            args.naive_bounds = compute_confidence_per_dim(solutions, args, 
                                                           [max(rho_list)])
            print('Naive bounds obtained by analyzing measures independently:', 
                  args.naive_bounds)
            
        scen_opt_time = time.process_time() - time_before_scenario
        
        print("\n===== Scenario problems completed at:", getTime(),"=====")
        
        # Plot results
        plot_results(output_path, args, regions, solutions)
        
        print("\n===== Plotting completed at:", getTime(),"=====")
        
        # Validation of results
        if args.Nvalidate > 0:
            # Sample new parameter for validation
            param_values = get_parameter_values(args.Nvalidate, param_dic)
            
            emp_satprob, val_dfs = \
                validate_solutions(val_dfs, sampler, regions, args.Nvalidate, 
                   properties, list(param_dic.keys()), param_values,
                   root_dir=root_dir, cache=args.modelfile_nosuffix+'_seed'+ \
                       str(args.seeds)+'_cache.pkl')
                                
            dfs['regions_stats']['Emp_satprob'] = emp_satprob
                                
        else:
            emp_satprob = [None]*len(regions)
            
        print("\n===== Validation completed at:", getTime(),"=====")
        
        dfs['storm_stats'] = print_stats(sampler.get_stats())
        
        # Add results to the DataFrame for exporting to the benchmark table
        if q == 0:
            STATS['Nr. props (Phi)'] = int(dfs['storm_stats']['no_properties'])
            STATS['#pars'] = int(dfs['storm_stats']['no_parameters'])
            STATS['#states'] = int(dfs['storm_stats']['orig_model_states'])
            STATS['#trans'] = int(dfs['storm_stats']['orig_model_transitions'])
            
            STATS['time_init'] = np.round(dfs['storm_stats']['time_load'] + \
                                         dfs['storm_stats']['time_bisim'], 2)
            STATS['time_sample (x100)'] = np.round(
                dfs['storm_stats']['time_sample'] / itr['N'] * 100, 2)
        STATS['scen_opt_N='+str(itr['N'])] = np.round(scen_opt_time, 2)
        
        # Save raw results in Excel file
        dfs['arguments'] = pd.Series(vars(args))
        save_results(output_path, dfs, args.modelfile_nosuffix, 
                     args.Nsamples)
    
        # Export validation results to csv (for easy plotting to paper)
        if args.export_bounds:
            cases.add_eta_row(q, args, regions, emp_satprob)
        if args.export_runtime:
            cases.add_time_row(args, scen_opt_time)
            cases.write_time()
        
    # Merge with the latest benchmark table csv export file
    if args.export_stats:
        export_benchmark_table(root_dir, args.export_stats, STATS)