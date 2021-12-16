from slurf.sample_cache import SampleCache
from slurf.approximate_ctmc_checker import ApproximateChecker

import stormpy as sp
import stormpy.pars
import stormpy.dft

import math
import time


class ModelSamplerInterface:
    """
    Describes the interface for sampling parametric models.
    """

    def __init__(self):
        self._model = None
        self._init_state = None
        self._properties = None
        self._parameters = None
        self._inst_checker_approx = None
        self._inst_checker_exact = None
        self._samples = None
        # Statistics
        self._states_orig = 0
        self._transitions_orig = 0
        self._time_load = 0
        self._time_bisim = 0
        self._time_sample = 0
        self._sample_calls = 0
        self._refined_samples = 0

    def load(self, model, properties):
        """

        Initialize sampler with model and properties.

        Parameters
        ----------
        model Description file for the (parametric) model.
        properties Properties here is either a tuple (event, [time bounds]) or a list of properties.

        Returns Dict of all parameters and their bounds (default bounds are [0, infinity)).
        """

    def sample(self, valuation):
        """
        Analyse the model according to a sample point.

        Parameters
        ----------
        valuation Parameter valuation in form of a dictionary from parameters to values.

        Returns
        -------
        Sample point containing the result of evaluating the model for each property.
        """

    def sample_batch(self, samples):
        """
        Analyse the model according to a batch of sample points.

        Parameters
        ----------
        samples List of samples to check.

        Returns
        -------
        A dictionary of results corresponding to the names of the sample points.
        """

    def refine(self, sample_id):
        """
        Refine sample point and obtain exact result.

        Parameters
        ----------
        sample_id Id of sample to refine.

        Returns
        -------
        Sample point containing the refined result.
        """

    def refine_batch(self, sample_ids):
        """
        Refine sample points and obtain exact result.

        Parameters
        ----------
        sample_ids Ids of samples to refine.

        Returns
        -------
        A dictionary of results corresponding to the names of the sample points.
        """

    def get_stats(self):
        """
        Get statistics on model and timings.

        Returns
        -------
        Dictionary  with interesting stats
        """
        return {
            "model_states": self._model.nr_states,
            "model_transitions:": self._model.nr_transitions,
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


class CtmcReliabilityModelSamplerInterface(ModelSamplerInterface):
    """
    This simple interface builds a parametric CTMC and then uses an instantiation checker to check the model.
    """

    def init_from_model(self, model):
        """
        Initialize sampler from CTMC model.

        Parameters
        ----------
        model CTMC.

        Returns Dict of all parameters and their bounds (default [0, infinity)).
        -------

        """
        self._model = model
        # Simplify model
        # Keep track of size of original model
        self._states_orig = self._model.nr_states
        self._transitions_orig = self._model.nr_transitions
        time_start = time.process_time()
        # Apply bisimulation minimisation
        self._model = sp.perform_bisimulation(self._model, self._properties, sp.BisimulationType.STRONG)
        self._time_bisim = time.process_time() - time_start

        # Get (unique) initial state
        assert len(self._model.initial_states) == 1
        self._init_state = self._model.initial_states[0]
        # Get parameters
        self._parameters = {p.name: p for p in self._model.collect_all_parameters()}

        # Create instantiation model checkers
        self._inst_checker_exact = sp.pars.PCtmcInstantiationChecker(self._model)
        self._inst_checker_approx = ApproximateChecker(self._model)

        # Create sample cache
        self._samples = SampleCache()
        # Return all parameters each with range (0 infinity)
        return {p: (0, math.inf) for p in self._parameters.keys()}

    def prepare_properties(self, properties, program=None):
        """
        Set properties.

        Parameters
        ----------
        properties Properties either given as a tuple (event, [time bounds]) or a list of properties.
        program Prism program (optional).
        -------

        """
        if isinstance(properties, list):
            # List of properties
            property_string = ";".join(properties)
        else:
            # Given as tuple (reachability label, [time bounds])
            event, time_bounds = properties[0], properties[1]
            property_string = ";".join([f'P=? [ F<={float(t)} "{event}" ]' for t in time_bounds])
        if program is not None:
            self._properties = sp.parse_properties_for_prism_program(property_string, program)
        else:
            self._properties = sp.parse_properties(property_string)

    def load(self, model, properties):
        """

        Initialize sampler with model and properties.

        Parameters
        ----------
        model A CTMC with a label.
        properties Properties here is either a tuple (event, [time bounds]) or a list of properties.

        Returns Dict of all parameters and their bounds (default [0, infinity)).
        -------

        """
        time_start = time.process_time()
        # Load prism program
        program = sp.parse_prism_program(model, prism_compat=True)
        # Create properties
        self.prepare_properties(properties, program)
        # Build (sparse) CTMC
        options = sp.BuilderOptions([p.raw_formula for p in self._properties])
        model = sp.build_sparse_parametric_model_with_options(program, options)
        parameters = self.init_from_model(model)
        self._time_load = time.process_time() - time_start
        return parameters

    def _sample(self, sample_point):
        # Create parameter valuation
        storm_valuation = {self._parameters[p]: sp.RationalRF(val) for p, val in sample_point.get_valuation().items()}

        # Analyse each property individually (Storm does not allow multiple properties for the InstantiationModelChecker
        results = []
        for prop in self._properties:
            # Specify formula
            self._inst_checker_approx.specify_formula(prop.raw_formula)
            # Check CTMC
            lb, ub = self._inst_checker_approx.check(storm_valuation)
            results.append((lb, ub))
        # Add result
        sample_point.set_results(results, refined=False)
        return sample_point

    def sample(self, valuation, exact=False):

        # Create sample point
        self.check_correct_valuation(valuation)
        sample_point = self._samples.add_sample(valuation)

        if exact:
            return self._refine(sample_point)

        time_start = time.process_time()
        sample_point = self._sample(sample_point)
        self._time_sample += time.process_time() - time_start
        self._sample_calls += 1
        return sample_point

    def sample_batch(self, samples, exact=False):
        # Create sample points
        sample_points = []
        for valuation in samples:
            self.check_correct_valuation(valuation)
            sample_points.append(self._samples.add_sample(valuation))

        if exact:
            # TODO: more efficient
            return self.refine_batch([s.get_id() for s in sample_points])

        time_start = time.process_time()
        results = dict()
        # TODO: use better approach than simply iterating
        for sample_point in sample_points:
            results[sample_point.get_id()] = self._sample(sample_point)

        self._time_sample += time.process_time() - time_start
        self._sample_calls += len(samples)
        return results

    def _refine(self, sample_point):
        assert not sample_point.is_refined()
        # Create parameter valuation
        storm_valuation = {self._parameters[p]: sp.RationalRF(val) for p, val in sample_point.get_valuation().items()}

        # Parameter valuation must be graph preserving
        self._inst_checker_exact.set_graph_preserving(True)

        env = sp.Environment()
        # Analyse each property individually (Storm does not allow multiple properties for the InstantiationModelChecker
        results = []
        for prop in self._properties:
            # Specify formula
            self._inst_checker_exact.specify_formula(sp.ParametricCheckTask(prop.raw_formula, True))  # Only initial states
            # Check CTMC
            results.append(self._inst_checker_exact.check(env, storm_valuation).at(self._init_state))
        # Add result
        sample_point.set_results(results, True)
        return sample_point

    def refine(self, sample_id):
        time_start = time.process_time()

        # Get corresponding sample point
        sample = self._samples.get_sample(sample_id)
        # Refine sample
        self._refine(sample)

        self._time_sample += time.process_time() - time_start
        self._refined_samples += 1
        return sample

    def refine_batch(self, sample_ids):
        time_start = time.process_time()

        # Get corresponding sample points
        samples = [self._samples.get_sample(sample_id) for sample_id in sample_ids]

        for sample in samples:
            # Refine sample
            self._refine(sample)

        self._time_sample += time.process_time() - time_start
        self._refined_samples += len(samples)
        return samples


class DftReliabilityModelSamplerInterface(CtmcReliabilityModelSamplerInterface):
    """
    This simple interface builds a parametric DFT, generates the corresponding parametric CTMC
    and then uses an instantiation checker to check the model.
    """

    def __init__(self):
        super(CtmcReliabilityModelSamplerInterface, self).__init__()
        self._dft = None

    def load(self, model, properties):
        """

        Parameters
        ----------
        model A DFT with parametric failure rates.
        properties Properties here is either a tuple (event, [time bounds]) or a list of properties.

        Returns Dictionary of parameters and their bounds.
        -------

        """
        time_start = time.process_time()
        # Load DFT from Galileo file
        self._dft = sp.dft.load_parametric_dft_galileo_file(model)

        # Create properties
        self.prepare_properties(properties)

        # Use the first property to generate the CTMC state space
        # Obtain CTMC by exporting to DRN format and loading again
        # TODO: implement dedicated methods
        drn_file = "tmp_ctmc.drn"
        sp.set_settings(["--io:exportexplicit", drn_file])
        tmp_prop = sp.parse_properties(f'T=? [ F "failed" ]')[0]
        stormpy.dft.analyze_parametric_dft(self._dft, [tmp_prop.raw_formula])
        parameters = self.init_from_model(sp.build_parametric_model_from_drn(drn_file))
        self._time_load = time.process_time() - time_start
        return parameters

    def get_stats(self):
        stats = super(CtmcReliabilityModelSamplerInterface, self).get_stats()
        stats["dft_be"] = self._dft.nr_be()
        stats["dft_elements"] = self._dft.nr_elements()
        stats["dft_dynamic"] = self._dft.nr_dynamic()
        return stats
