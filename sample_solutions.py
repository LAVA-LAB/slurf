import numpy as np
import os

from slurf.model_sampler_interface import \
    CtmcReliabilityModelSamplerInterface
from slurf.sample_cache import SampleCache, import_sample_cache, \
    export_sample_cache

def param_SIR(Nsamples):
    
    param_values = np.random.uniform(low=[0.05, 0.05], 
                                     high=[0.08, 0.08],
                                     size=(Nsamples, 2))
    
    return param_values

def param_SIR_gauss(Nsamples):
    
    param_values = np.random.multivariate_normal(
                        mean = np.array([0.065, 0.065]),
                        cov = np.diag([0.0001, 0.0001]),
                        size=(Nsamples))
    
    param_values = np.maximum(param_values, 1e-3)
    
    return param_values

def sample_solutions(Nsamples, model, properties, root_dir, cache=False):
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

    param_list = ['ki', 'kr']
    results = np.zeros((Nsamples, len(properties[1])))
    
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

    # If the script has not returned yet, we load samples
    parameters = param_SIR(Nsamples)
    
    assert len(param_list) == parameters.shape[1]

    parameters_dic = [{param: row[i] for i,param in enumerate(param_list)} 
                      for row in parameters]

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
