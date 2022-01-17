import numpy as np
import os
import pandas as pd

from slurf.sample_cache import SampleCache, import_sample_cache, \
    export_sample_cache
from slurf.commons import path

def load_distribution(root_dir, model_path, model_type,
                      parameters_file=None, properties_file=None):
    """
    Helper function to load probability distribution and parameter data
    :model_folder: Subfolder to load the model from
    :model_file: Filename of the model to load
    """
    
    if not parameters_file:
        parameters_file = 'parameters.xlsx'
    if not properties_file:
        properties_file = 'properties.xlsx'
    
    # Split path between (sub)folder and filename
    model_folder, model_file = model_path.rsplit('/', 1)
    
    param_path = path(root_dir, "models/"+str(model_folder), parameters_file)
    try:
        open(param_path)
    except IOError:
        print("ERROR: Parameter distribution file (named 'parameters.xlsx') does not exist")
        
    model_file = path(root_dir, "models/"+str(model_folder), model_file)
    try:
        open(model_file)
    except IOError:
        print("ERROR: Model file does not exist")
    
    # Read parameter sheet
    param_df = pd.read_excel(param_path, sheet_name='Parameters', index_col=0)
    param_dic = param_df.to_dict('index')

    properties_path = path(root_dir, "models/"+str(model_folder), properties_file)

    # Interpretation of properties differs between CTMCs and DFTs
    if model_type == 'CTMC':
        
        # Read property sheet
        property_df = pd.read_excel(properties_path, sheet_name='Properties')
        property_df = property_df[ property_df['enabled'] == True ]
        properties = property_df['property'].to_list()
        prop_labels = property_df['label'].to_list()
        
        if 'time' in property_df:
            reliability = True
            timebounds = property_df['time'].to_list()
        else:
            reliability = False
            timebounds = None
            
    else:
        
        # Read property sheet
        property_df = pd.read_excel(properties_path, sheet_name='Properties')
        
        timebounds = tuple(property_df['failed'])        
        properties = ("failed", timebounds)
        
        prop_labels = ["failed[t<="+str(t)+"]" for t in timebounds]
        reliability = True
    
    return model_file, param_dic, properties, prop_labels, reliability, timebounds


def get_parameter_values(Nsamples, param_dic):
    
    # Initialize parameter matrix
    param_matrix = np.zeros((Nsamples, len(param_dic)))
    
    # For every parameter
    for i, (param_key, v) in enumerate(param_dic.items()):
        
        assert 'type' in v
        
        # Get Nsamples values
        if v['type'] == 'interval':
            assert 'lb' in v
            assert 'ub' in v
            
            param_matrix[:, i] = param_interval(Nsamples, v['lb'], v['ub'])
            
        elif v['type'] == 'gaussian':
            assert 'mean' in v
            assert 'std' in v
            
            param_matrix[:, i] = param_gaussian(Nsamples, v['mean'], 
                                                v['std'])
            
        # If parameter values are defined as 1/value, than inverse them
        if 'inverse' in v:
            if v['inverse'] is True:
                param_matrix[:, i] = 1/param_matrix[:, i]
            
    return param_matrix
            

def param_interval(Nsamples, lb, ub):
    
    param_values = np.random.uniform(low=lb, high=ub, size=Nsamples)
    
    return param_values


def param_gaussian(Nsamples, mean, std):
    
    param_values = np.random.normal(loc=mean, scale=std, size=Nsamples)
    
    # Make values nonnegative
    param_values = np.maximum(param_values, 1e-9)
    
    return param_values


def sample_solutions(sampler, Nsamples, properties, param_list, 
                     param_values, root_dir=None, cache=False, exact=False):
    """

    Parameters
    ----------
    sampler Sampler object (for either CTMC of DFT)
    Nsamples Number of samples
    properties List of property strings
    param_list Names (labels) of the parameters
    param_values List of values for every parameter
    root_dir Root directory where script is being run
    cache If True, we export the samples to a cache file

    Returns 2D Numpy array with every row being a sample
    -------

    """
    
    if type(properties) == tuple:
        num_props = len(properties[1])
    else:
        num_props = len(properties)
        
    # If we compute imprecise solutions, results array is 3D (to store upp/low)
    if exact:
        results = np.zeros((Nsamples, num_props))
    else:
        results = np.zeros((Nsamples, num_props, 2))
    
    # If cache is activated...
    if cache:
        cache_path = os.path.join(root_dir, 'cache', cache)
        
        # If cache file exists, try to load samples from it
        if os.path.isfile(cache_path):
        
            # Import cache
            samples_imp = import_sample_cache(cache_path)
            
            # Test if the parameters match
            params_import = list(samples_imp.get_sample(0).get_valuation().keys())
            num_properties = len(samples_imp.get_sample(0).get_result())
            
            # Only interpret results if the cache file matches the current call
            if params_import != param_list:
                print('- Cache incompatible: number of parameters does not match')
            elif Nsamples != samples_imp.num_samples:
                print('- Cache incompatible: number of samples does not match')
            elif num_props != num_properties:
                print('- Cache incompatible: number of properties does not match')
            else:
                
                for n in range(Nsamples):
                    results[n] = samples_imp.get_sample(n).get_result()
                    
                print('- Samples imported from cache')
                print('--- List of parameters:',params_import)
                print('--- Number of samples:',samples_imp.num_samples)
                print('--- Number of properties:',num_properties)
                
                # If results are imported, return here
                return results
    
    assert len(param_list) == param_values.shape[1]

    parameters_dic = [{param: row[i] for i,param in enumerate(param_list)} 
                      for row in param_values]

    sampleObj = sampler.sample_batch(parameters_dic, exact)

    for n in range(Nsamples):

        results[n] = sampleObj[n].get_result()
        
        if not exact:
            if all(np.isclose(results[n,:,0], results[n,:,1])):
                sampleObj[n]._refined = True
                print('Sample',n,'is already precise!')
        
    # If cache is activated, export samples
    if cache:
        # Initialize cache
        sample_cache = SampleCache()
        
        cache_samples = {}
        for n in range(Nsamples):
            cache_samples[n] = sample_cache.add_sample(parameters_dic[n])
            cache_samples[n].set_results(results[n])
        
        export_sample_cache(sample_cache, cache_path)
        print('- Samples exported to cache')

    return sampleObj, results


def refine_solutions(sampler, sampleObj, solutions, idx):
    '''
    Refine imprecise solutions for the given indices

    Parameters
    ----------
    sampler Sampler object (for either CTMC of DFT)
    solutions 3D np.array with solutions
    idx List of indices to refine
    
    Returns
    -------
    solutions Updated solutions array
    '''
    
    samples = sampler.refine_batch(idx)
    for i, n in enumerate(idx):
        sol = samples[i].get_result()
        solutions[n,:,0] = sol
        solutions[n,:,1] = sol
        
        sampleObj[n] = samples[i]
    
    return solutions


def validate_solutions(val_dfs, sampler, regions, Nvalidate, properties, 
                       param_list, param_values, root_dir=None, cache=None):
    """

    Parameters
    ----------
    val_dfs DataFrame where the validation results are stored over repetitions
    sampler Sampler object (for either CTMC of DFT)
    regions Solutions to scenario problems (for multiple values of rho)
    Nvaliate Number of samples
    properties List of property strings
    param_list Names (labels) of the parameters
    param_values List of values for every parameter

    NOTE: Validation currently only works for rectangular confidence regions

    Returns the empirical violation probability for every value of rho
    -------

    """
    
    # Compute new solutions for validation
    solutions_V = sample_solutions( sampler = sampler,
                        Nsamples = Nvalidate,
                        properties = properties,
                        param_list = param_list,
                        param_values = param_values,
                        root_dir = root_dir,
                        cache = cache )
    
    # Initialize the empirical violation probability
    empirical_satprob  = np.zeros(len(regions))
    
    # For every value of rho
    for i,D in regions.items():
        # Determine which samples violate the solution (bounding box)
        sat_low = np.all(solutions_V >= D['x_low'], axis=1)
        sat_upp = np.all(solutions_V <= D['x_upp'], axis=1)
        
        # Compute the total number of violating samples
        sat_sum = np.sum(sat_low * sat_upp)
        
        # Compute the empirical violation probability 
        empirical_satprob[i] = sat_sum / Nvalidate
        
    for i, item in regions.items():
        
        rho = np.round(item['rho'], 4)
        series = regions[i]['eta_series']
        series['emp'] = empirical_satprob[i]
        
        if rho in val_dfs:
            val_dfs[rho] = val_dfs[rho].append(series, ignore_index=True)
        else:
            val_dfs[rho] = pd.DataFrame(series).T

    return empirical_satprob, val_dfs