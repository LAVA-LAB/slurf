import numpy as np
import os

from slurf.model_sampler_interface import \
    CtmcReliabilityModelSamplerInterface
from slurf.sample_cache import SampleCache, import_sample_cache, \
    export_sample_cache

def get_parameter_values(Nsamples, param_dic):
    
    # Initialize parameter matrix
    param_matrix = np.zeros((Nsamples, len(param_dic)))
    
    # For every parameter
    for i, (param_key, v) in enumerate(param_dic.items()):
        
        assert 'type' in v
        
        # Get Nsamples values
        if v['type'] == 'interval':
            assert 'lower_bound' in v
            assert 'upper_bound' in v
            
            param_matrix[:, i] = param_interval(Nsamples, v['lower_bound'], 
                                                v['upper_bound'])
            
        elif v['type'] == 'gaussian':
            assert 'mean' in v
            assert 'covariance' in v
            if not 'nonzero' in v:
                v['nonzero'] = True
            
            param_matrix[:, i] = param_gaussian(Nsamples, v['mean'], 
                                                v['covariance'], v['nonzero'])
            
    return param_matrix
            
            
        

def param_interval(Nsamples, lb, ub):
    
    param_values = np.random.uniform(low=lb, high=ub, size=Nsamples)
    
    return param_values

def param_gaussian(Nsamples, mean, cov, nonzero=True):
    
    param_values = np.random.normal(loc=mean, scale=cov, size=Nsamples)
    
    if nonzero:
        param_values = np.maximum(param_values, 1e-3)
    
    return param_values

def sample_solutions(Nsamples, model, properties, param_list, param_values,
                     root_dir, cache=False):
    """

    Parameters
    ----------
    Nsamples Number of samples
    model File name of the model to load
    properties List of property strings
    cache If True, we export the samples to a cache file

    Returns 2D Numpy array with every row being a sample
    -------

    """

    # Load model
    sampler = CtmcReliabilityModelSamplerInterface()
    sampler.load(model, properties)
    
    if type(properties) == tuple:
        num_props = len(properties[1])
    else:
        num_props = len(properties)
        
    results = np.zeros((Nsamples, num_props))
    
    cache_file = os.path.join(root_dir, "samples.pkl")
    
    # If cache is activated and cache file exists, try to load samples from it
    if cache and os.path.isfile(cache_file):
        
        # Import cache
        samples_imp = import_sample_cache(cache_file)
        
        # Test if the parameters match
        params_import = list(samples_imp.get_sample(0).get_valuation().keys())
        num_properties = len(samples_imp.get_sample(0).get_result())
        
        # Only interpret results if the cache file matches the current call
        # if model != samples_imp.model:
        #     print('- Cache incompatible:')
        #     print('--- Model specified is: "'+str(model)+'"')
        #     print('--- Model in cachce is: "'+str(samples_imp.model)+'"')
        if params_import != param_list:
            print('- Cache incompatible: number of parameters does not match')
        elif Nsamples != samples_imp.num_samples:
            print('- Cache incompatible: number of samples does not match')
        elif len(properties[1]) != num_properties:
            print('- Cache incompatible: number of properties does not match')
        else:
            
            for n in range(Nsamples):
                results[n] = samples_imp.get_sample(n).get_result()
                
            print('- Samples imported from cache')
            print('--- List of parameters:',params_import)
            print('--- Number of samples:',samples_imp.num_samples)
            print('--- Number of properties:',num_properties)
            
            # If results are imported, return here
            return sampler, results
    
    assert len(param_list) == param_values.shape[1]

    parameters_dic = [{param: row[i] for i,param in enumerate(param_list)} 
                      for row in param_values]

    sampleIDs = sampler.sample_batch(parameters_dic, exact=True)

    for n in range(Nsamples):

        results[n] = sampleIDs[n].get_result()
        
    # If cache is activated, export samples
    if cache:
        # Initialize cache
        sample_cache = SampleCache()
        
        cache_samples = {}
        for n in range(Nsamples):
            cache_samples[n] = sample_cache.add_sample(parameters_dic[n])
            cache_samples[n].set_results(results[n])
        
        export_sample_cache(sample_cache, cache_file)
        print('- Samples exported to cache')

    return sampler, results
