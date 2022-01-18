from slurf.sample_cache import SampleCache
from slurf.approximate_ctmc_checker import ApproximationOptions

import time
from tqdm import tqdm


class ModelSamplerInterface:
    """
    Describes the general interface for sampling parametric models.
    """

    def __init__(self):
        self._model = None
        self._symb_desc = None
        self._init_state = None
        self._properties = None
        self._parameters = None
        self._inst_checker_approx = None
        self._approx_options = ApproximationOptions()
        self._inst_checker_exact = None
        self._samples = SampleCache()

        # Statistics
        self._states_orig = 0
        self._transitions_orig = 0
        self._time_load = 0
        self._time_bisim = 0
        self._time_sample = 0
        self._sample_calls = 0
        self._refined_samples = 0

    def set_max_cluster_distance(self, max_distance):
        self._approx_options.set_cluster_max_distance(max_distance)

    def set_approximation_heuristic(self, approx_heuristic):
        self._approx_options.set_approx_heuristic(approx_heuristic)

    def load(self, model, properties, bisim=True, constants=None):
        """

        Initialize sampler with model and properties.

        Parameters
        ----------
        model Description file for the (parametric) model.
        properties Properties here is either a tuple (event, [time bounds]) or a list of properties.
        bisim Whether to apply bisimulation.
        constants Constants for graph changing variables in model description (optional)

        Returns Dict of all parameters and their bounds (default bounds are [0, infinity)).
        """

    def _sample(self, sample_point):
        pass

    def sample(self, valuation, exact=False):
        """
        Analyse the model according to a sample point.

        Parameters
        ----------
        valuation Parameter valuation in form of a dictionary from parameters to values.
        exact Whether exact results should be computed.

        Returns
        -------
        Sample point containing the result of evaluating the model for each property.
        """
        # Create sample point
        self.check_correct_valuation(valuation)
        sample_point = self._samples.add_sample(valuation)

        if exact:
            return self._refine(sample_point, precision=0)

        time_start = time.process_time()
        sample_point = self._sample(sample_point)
        self._time_sample += time.process_time() - time_start
        self._sample_calls += 1
        return sample_point

    def sample_batch(self, samples, exact=False):
        """
        Analyse the model according to a batch of sample points.

        Parameters
        ----------
        samples List of samples to check.
        exact Whether exact results should be computed.

        Returns
        -------
        A dictionary of results corresponding to the names of the sample points.
        """
        # Create sample points
        sample_points = []
        for valuation in samples:
            self.check_correct_valuation(valuation)
            sample_points.append(self._samples.add_sample(valuation))

        if exact:
            # TODO: more efficient
            return self.refine_batch([s.get_id() for s in sample_points], precision=0)

        time_start = time.process_time()
        results = dict()
        # TODO: use better approach than simply iterating
        for sample_point in tqdm(sample_points):
            results[sample_point.get_id()] = self._sample(sample_point)

        self._time_sample += time.process_time() - time_start
        self._sample_calls += len(samples)
        return results

    def _refine(self, sample_point, precision, ind_precisions=dict()):
        pass

    def refine(self, sample_id, precision, ind_precisions=dict()):
        """
        Refine sample point and obtain exact result.

        Parameters
        ----------
        sample_id Id of sample to refine.
        precision Maximal allowed distance between upper and lower bound.
        ind_precisions Dictionary with individual precisions for given properties. If property is not given,
            the default precision is used.

        Returns
        -------
        Sample point containing the refined result.
        """
        time_start = time.process_time()

        # Get corresponding sample point
        sample = self._samples.get_sample(sample_id)
        # Refine sample
        self._refine(sample, precision, ind_precisions)

        self._time_sample += time.process_time() - time_start
        self._refined_samples += 1
        return sample

    def refine_batch(self, sample_ids, precision, ind_precisions=dict()):
        """
        Refine sample points and obtain exact result.

        Parameters
        ----------
        sample_ids Ids of samples to refine.
        precision Maximal allowed distance between upper and lower bound.
        ind_precisions Dictionary with individual precisions for given properties. If property is not given,
            the default precision is used.
        Returns
        -------
        A dictionary of results corresponding to the names of the sample points.
        """
        time_start = time.process_time()

        # Get corresponding sample points
        samples = [self._samples.get_sample(sample_id) for sample_id in sample_ids]

        for sample in tqdm(samples):
            # Refine sample
            self._refine(sample, precision, ind_precisions)

        self._time_sample += time.process_time() - time_start
        self._refined_samples += len(samples)
        return samples

    def get_stats(self):
        """
        Get statistics on model and timings.

        Returns
        -------
        Dictionary  with interesting stats
        """
        return {
            "model_states": self._model.nr_states,
            "model_transitions": self._model.nr_transitions,
            "orig_model_states": self._states_orig,
            "orig_model_transitions": self._transitions_orig,
            "no_parameters": len(self._parameters),
            "no_samples": len(self._samples.get_samples()),
            "no_properties": len(self._properties),
            "sample_calls": self._sample_calls,
            "refined_samples": self._refined_samples,
            "time_load": round(self._time_load, 4),
            "time_bisim": round(self._time_bisim, 4),
            "time_sample": round(self._time_sample, 4)
        }

    def check_correct_valuation(self, valuation):
        # Check that a valuation instantiates all parameters
        if len(valuation) != len(self._parameters):
            print("ERROR: not all parameters are instantiated by valuation")
            assert False
        for param in valuation.keys():
            if param not in self._parameters:
                print("ERROR: parameter {} is not given in valuation".format(param))
                assert False
