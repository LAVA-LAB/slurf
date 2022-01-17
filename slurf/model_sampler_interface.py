from slurf.sample_cache import SampleCache
from slurf.approximate_ctmc_checker import ApproximateChecker, ApproximationOptions
import slurf.util as util

import stormpy as sp
import stormpy.pars
import stormpy.dft

import math
import os.path
import time
from tqdm import tqdm


class ModelSamplerInterface:
    """
    Describes the interface for sampling parametric models.
    """

    def __init__(self):
        self._model = None
        self._symb_desc = None
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


class CtmcReliabilityModelSamplerInterface(ModelSamplerInterface):
    """
    This simple interface builds a parametric CTMC and then uses an instantiation checker to check the model.
    """

    def init_from_model(self, model, bisim=True):
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
        if bisim:
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

    def prepare_properties(self, properties):
        """
        Set properties.

        Parameters
        ----------
        properties Properties either given as a tuple (event, [time bounds]) or a list of properties.
        -------

        """
        if isinstance(properties, list):
            # List of properties
            property_string = ";".join(properties)
        else:
            # Given as tuple (reachability label/expression, [time bounds])
            event, time_bounds = properties[0], properties[1]
            # Todo: use less hackish way to distinguish between expression and label
            ev_str = event if "=" in event else f'"{event}"'
            property_string = ";".join([f'P=? [ F<={float(t)} {ev_str} ]' for t in time_bounds])
        if self._symb_desc is not None:
            if self._symb_desc.is_prism_program:
                self._properties = sp.parse_properties_for_prism_program(property_string, self._symb_desc.as_prism_program())
            else:
                assert self._symb_desc.is_jani_model
                self._properties = sp.parse_properties_for_jani_model(property_string, self._symb_desc.as_jani_model())
        else:
            self._properties = sp.parse_properties(property_string)

    def load(self, model, properties, bisim=True, constants=None):
        """

        Initialize sampler with model and properties.

        Parameters
        ----------
        model A CTMC with a label.
        properties Properties here is either a tuple (event, [time bounds]) or a list of properties.
        bisim Whether to apply bisimulation.
        constants Constants for graph changing variables in model description (optional)

        Returns Dict of all parameters and their bounds (default [0, infinity)).
        -------

        """
        time_start = time.process_time()
        jani_file = (os.path.splitext(model)[1] == ".jani")

        if jani_file:
            # Parse Jani program
            model_desc, formulas = stormpy.parse_jani_model(model)
            if constants:
                symb_desc = stormpy.SymbolicModelDescription(model_desc)
                constant_definitions = symb_desc.parse_constant_definitions(constants)
                model_desc = symb_desc.instantiate_constants(constant_definitions).as_jani_model()
        else:
            # Parse Prism program
            model_desc = sp.parse_prism_program(model, prism_compat=True)
        self._symb_desc = stormpy.SymbolicModelDescription(model_desc)
        # Create properties
        self.prepare_properties(properties)
        # Build (sparse) CTMC
        options = sp.BuilderOptions([p.raw_formula for p in self._properties])
        model = sp.build_sparse_parametric_model_with_options(model_desc, options)
        parameters = self.init_from_model(model, bisim)
        self._time_load = time.process_time() - time_start
        return parameters

    def _sample(self, sample_point):
        # Create parameter valuation
        storm_valuation = {self._parameters[p]: sp.RationalRF(val) for p, val in sample_point.get_valuation().items()}

        # Analyse each property individually (Storm does not allow multiple properties for the InstantiationModelChecker
        results = []
        for prop in self._properties:
            # Specify formula
            self._inst_checker_approx.specify_formula(prop.raw_formula, self._symb_desc)
            # Check CTMC
            lb, ub = self._inst_checker_approx.check(storm_valuation, sample_point.get_id())
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
        for sample_point in tqdm(sample_points):
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

        for sample in tqdm(samples):
            # Refine sample
            self._refine(sample)

        self._time_sample += time.process_time() - time_start
        self._refined_samples += len(samples)
        return samples


class DftModelSamplerInterface(CtmcReliabilityModelSamplerInterface):
    """
    General class for DFT sampler interface.
    """

    def __init__(self):
        super(CtmcReliabilityModelSamplerInterface, self).__init__()
        self._dft = None

    def get_stats(self):
        stats = super(CtmcReliabilityModelSamplerInterface, self).get_stats()
        stats["dft_be"] = self._dft.nr_be()
        stats["dft_elements"] = self._dft.nr_elements()
        stats["dft_dynamic"] = self._dft.nr_dynamic()
        return stats


class DftParametricModelSamplerInterface(DftModelSamplerInterface):
    """
    This simple interface builds a parametric DFT, generates the corresponding parametric CTMC
    and then uses an instantiation checker to check the model.
    """

    def load(self, model, properties, bisim=True, constants=None):
        """

        Parameters
        ----------
        model A DFT with parametric failure rates.
        properties Properties here is either a tuple (event, [time bounds]) or a list of properties.
        bisim Whether to apply bisimulation.
        constants Constants for graph changing variables in model description (not required for fault trees)

        Returns Dictionary of parameters and their bounds.
        -------

        """
        time_start = time.process_time()
        print(' - Load DFT from Galileo file')
        # Load DFT from Galileo file
        self._dft = sp.dft.load_parametric_dft_galileo_file(model)
        # Make DFT well-formed
        self._dft = sp.dft.transform_dft(self._dft, unique_constant_be=True, binary_fdeps=True)
        # Check for dependency conflicts -> no conflicts mean CTMC
        sp.dft.compute_dependency_conflicts(self._dft, use_smt=False, solver_timeout=0)

        # Create properties
        self.prepare_properties(properties)

        # Build CTMC from DFT
        print(' - Start building state space')
        # Set empty symmetry as rates can change which destroys symmetries
        empty_sym = sp.dft.DFTSymmetries()
        model = sp.dft.build_model(self._dft, empty_sym)
        print(' - Finished building model')

        if model.model_type == sp.ModelType.MA:
            print("ERROR: Resulting model is MA instead of CTMC")
            assert False

        parameters = self.init_from_model(model, bisim=bisim)
        self._time_load = time.process_time() - time_start
        return parameters


class DftConcreteApproximationSamplerInterface(DftModelSamplerInterface):
    """
    The approximation sampler does not build one parametric model but a partial model for each parameter valuation.
    Refinement can be done by exploring more of the state space.
    """

    def load(self, model, properties, bisim=True, constants=None):
        """
        Note that load() only constructs the DFT. The state space is built for each sample individually.

        Parameters
        ----------
        model A DFT with parametric failure rates.
        properties Properties here is either a tuple (event, [time bounds]) or a list of properties.
        bisim Whether to apply bisimulation.
        constants Constants for graph changing variables in model description (not required for fault trees)

        Returns Dictionary of parameters and their bounds.
        -------

        """
        time_start = time.process_time()
        print(' - Load DFT from Galileo file')
        # Load DFT from Galileo file
        self._dft = sp.dft.load_parametric_dft_galileo_file(model)
        # Make DFT well-formed
        self._dft = sp.dft.transform_dft(self._dft, unique_constant_be=True, binary_fdeps=True)
        # Check for dependency conflicts -> no conflicts mean CTMC
        sp.dft.compute_dependency_conflicts(self._dft, use_smt=False, solver_timeout=0)

        self._inst_checker_approx = sp.dft.DFTInstantiator(self._dft)

        # Create properties
        self.prepare_properties(properties)

        # Create sample cache
        self._samples = SampleCache()

        # Get parameters
        self._parameters = {p.name: p for p in sp.dft.get_parameters(self._dft)}
        self._time_load = time.process_time() - time_start
        return self._parameters

    def _sample(self, sample_point):
        # Create parameter valuation
        storm_valuation = {self._parameters[p]: sp.RationalRF(val) for p, val in sample_point.get_valuation().items()}

        # Instantiate parametric DFT
        sample_dft = self._inst_checker_approx.instantiate(storm_valuation)

        # Compute approximation from DFT
        print(' - Start building state space')
        builder = stormpy.dft.ExplicitDFTModelBuilder_double(sample_dft, sample_dft.symmetries())

        it = 0
        while True:
            builder.build(it, 1.0)
            self._model = builder.get_partial_model(True, False)
            self._init_state = self._model.initial_states[0]
            if self._model.model_type == sp.ModelType.MA:
                print("ERROR: Resulting model is MA instead of CTMC")
                assert False
            it += 1
            # TODO find better stopping criteria
            if self._model.nr_states > 1000 or it >= 3:
                break
        print(' - Finished building model')

        model_up = builder.get_partial_model(False, False)
        init_up = model_up.initial_states[0]
        results = []
        for prop in self._properties:
            result_low = stormpy.model_checking(self._model, prop).at(self._init_state)
            result_up = stormpy.model_checking(model_up, prop).at(init_up)
            results.append((result_low, result_up))

        sample_point.set_results(results, refined=False)
        return sample_point

    def _refine(self, sample_point):
        assert not sample_point.is_refined()
        # Create parameter valuation
        storm_valuation = {self._parameters[p]: sp.RationalRF(val) for p, val in sample_point.get_valuation().items()}

        # Instantiate parametric DFT
        sample_dft = self._inst_checker_approx.instantiate(storm_valuation)

        # Build CTMC from DFT
        print(' - Start building state space')
        self._model = sp.dft.build_model(sample_dft, sample_dft.symmetries())
        self._init_state = self._model.initial_states[0]
        print(' - Finished building model')

        if self._model.model_type == sp.ModelType.MA:
            print("ERROR: Resulting model is MA instead of CTMC")
            assert False

        results = []
        for prop in self._properties:
            # Check CTMC
            results.append(sp.model_checking(self._model, prop).at(self._init_state))

        sample_point.set_results(results, refined=True)
        return sample_point


class DftParametricApproximationSamplerInterface(DftParametricModelSamplerInterface):
    """
    The approximation sampler builds the complete parametric model and tries to use only partial models for sampling.
    """

    def __init__(self, cluster_max_distance):
        super(DftParametricModelSamplerInterface, self).__init__()
        self._clusters = dict()
        self._cluster_max_distance = cluster_max_distance

    def _sample(self, sample_point):
        # Create parameter valuation
        storm_valuation = {self._parameters[p]: sp.RationalRF(val) for p, val in sample_point.get_valuation().items()}

        # Find possible cluster
        absorbing_states = self._find_cluster(sample_point)
        if absorbing_states is None:
            # Compute states to remove
            absorbing_states = self._compute_absorbing_states(storm_valuation)
            print("  - Use new cluster")
            # Store cluster
            self._clusters[sample_point] = absorbing_states
        else:
            print("  - Use existing cluster")

        options = ApproximationOptions()
        options.set_fixed_states_absorbing(absorbing_states)
        self._inst_checker_approx = ApproximateChecker(self._model, options)

        # Analyse each property individually (Storm does not allow multiple properties for the InstantiationModelChecker
        results = []
        for prop in self._properties:
            # Specify formula
            self._inst_checker_approx.specify_formula(prop.raw_formula)
            # Check CTMC
            lb, ub = self._inst_checker_approx.check(storm_valuation, sample_point.get_id())
            assert util.leq(lb, ub)
            results.append((lb, ub))
        # Add result
        sample_point.set_results(results, refined=False)
        return sample_point

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

    def _find_cluster(self, sample_point):
        for point, absorbing in self._clusters.items():
            distance = sample_point.get_distance(point)
            if distance <= self._cluster_max_distance:
                # Found nearby cluster
                return absorbing
        # No cluster is close enough
        return None

    def _compute_absorbing_states(self, storm_valuation):
        # Check expected time on CTMC to as heuristic for important/unimportant states
        prop = sp.parse_properties('T=? [F "failed"]')[0]

        # Check CTMC
        self._inst_checker_exact.specify_formula(sp.ParametricCheckTask(prop.raw_formula, False))  # Get result for all states
        env = sp.Environment()
        result = self._inst_checker_exact.check(env, storm_valuation)
        assert result.result_for_all_states

        # Compute absorbing states
        result = dict(zip(range(self._model.nr_states), result.get_values()))
        res_reference = result[self._init_state] / 2.0
        return [i for i, res in result.items() if res < res_reference]

    def get_stats(self):
        stats = super(DftParametricModelSamplerInterface, self).get_stats()
        stats["no_approx_clusters"] = len(self._clusters)
        return stats
