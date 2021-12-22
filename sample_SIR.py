import numpy as np

from slurf.model_sampler_interface import \
    CtmcReliabilityModelSamplerInterface


def sample_SIR(Nsamples, Tlist, model):
    """

    Parameters
    ----------
    Nsamples Number of samples
    Tlist List of time points to evaluate at
    model File name of the model to load

    Returns 2D Numpy array with every row being a sample
    -------

    """

    # Load model
    sampler = CtmcReliabilityModelSamplerInterface()
    parameters_with_bounds = sampler.load(model, ("done", Tlist))

    results = np.zeros((Nsamples, len(Tlist)))
    parameters = np.random.uniform(low=[0.05, 0.05], high=[0.08, 0.08],
                                   size=(Nsamples, 2))

    parameters_dic = [{"ki": row[0], "kr": row[1]} for row in parameters]

    sampleIDs = sampler.sample_batch(parameters_dic, exact=True)

    for n in range(Nsamples):

        results[n] = sampleIDs[n].get_result()

    return sampler, sampleIDs, results
