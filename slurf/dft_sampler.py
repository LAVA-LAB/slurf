from slurf.ctmc_sampler import CtmcReliabilityModelSamplerInterface
from slurf.sample_cache import SampleCache
from slurf.model_sampler_interface import get_timebounds_and_target
import slurf.util

import stormpy as sp
import stormpy.pars
import stormpy.dft

import time
import random


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

    def __init__(self, all_relevant=False):
        super(DftModelSamplerInterface, self).__init__()
        self._all_relevant = all_relevant

    def load(self, model, properties, bisim=True, constants=None):
        """

        Parameters
        ----------
        :model: A DFT with parametric failure rates.
        :properties: Properties here is either a tuple (event, [time bounds]) or a list of properties.
        :bisim: Whether to apply bisimulation.
        :constants: Constants for graph changing variables in model description (not required for fault trees)

        Returns Dictionary of parameters and their bounds.
        -------

        """
        time_start = time.process_time()
        print(' - Load DFT from Galileo file')
        # Load DFT from Galileo file
        self._dft = sp.dft.load_parametric_dft_galileo_file(model)
        # Make DFT well-formed
        self._dft = sp.dft.transform_dft(self._dft, unique_constant_be=True, binary_fdeps=True, exponential_distributions=True)
        # Check for dependency conflicts -> no conflicts mean CTMC
        sp.dft.compute_dependency_conflicts(self._dft, use_smt=False, solver_timeout=0)

        # Create properties
        self.prepare_properties(properties)

        # Build CTMC from DFT
        print(' - Start building state space')
        # Set empty symmetry as rates can change which destroys symmetries
        empty_sym = sp.dft.DFTSymmetries()
        if self._all_relevant:
            relevant_events = sp.dft.compute_relevant_events(self._dft, [], ["all"])
            model = sp.dft.build_model(self._dft, empty_sym, relevant_events)
        else:
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

    def __init__(self, all_relevant=False):
        super(DftModelSamplerInterface, self).__init__()
        self._builders = dict()
        self._all_relevant = all_relevant

    def load(self, model, properties, bisim=True, constants=None):
        """
        Note that load() only constructs the DFT. The state space is built for each sample individually.

        Parameters
        ----------
        :model: A DFT with parametric failure rates.
        :properties: Properties here is either a tuple (event, [time bounds]) or a list of properties.
        :bisim: Whether to apply bisimulation.
        :constants: Constants for graph changing variables in model description (not required for fault trees)

        Returns Dictionary of parameters and their bounds.
        -------

        """
        time_start = time.process_time()
        print(' - Load DFT from Galileo file')
        # Load DFT from Galileo file
        self._dft = sp.dft.load_parametric_dft_galileo_file(model)
        # Make DFT well-formed
        self._dft = sp.dft.transform_dft(self._dft, unique_constant_be=True, binary_fdeps=True, exponential_distributions=True)
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
        sp.dft.compute_dependency_conflicts(sample_dft, use_smt=False, solver_timeout=0)  # Needed as instantiation looses this information

        # Create new builder
        assert sample_point not in self._builders

        if self._all_relevant:
            empty_sym = sp.dft.DFTSymmetries()
            builder = stormpy.dft.ExplicitDFTModelBuilder_double(sample_dft, empty_sym)
        else:
            builder = stormpy.dft.ExplicitDFTModelBuilder_double(sample_dft, sample_dft.symmetries())
        iteration = 0

        # Refine approximation from DFT
        print(' - Start refining state space')
        iterating = True
        while iterating:
            builder.build(iteration, 1.0, sp.dft.ApproximationHeuristic.PROBABILITY)
            self._model = builder.get_partial_model(True, False)
            self._init_state = self._model.initial_states[0]
            if self._model.model_type == sp.ModelType.MA:
                print("ERROR: Resulting model is MA instead of CTMC")
                assert False
            # TODO find better stopping criteria
            if self._model.nr_states > 1000 or iteration >= 5:
                iterating = False
            else:
                iteration += 1
        print(' - Finished building model after iteration {}'.format(iteration))
        # Store builder in cache
        self._builders[sample_point] = (builder, iteration)

        model_up = builder.get_partial_model(False, False)
        init_up = model_up.initial_states[0]
        results = []
        for prop in self._properties:
            result_low = stormpy.model_checking(self._model, prop).at(self._init_state)
            result_up = stormpy.model_checking(model_up, prop).at(init_up)
            results.append((result_low, result_up))

        sample_point.set_results(results, refined=False)
        return sample_point

    def _refine(self, sample_point, precision, ind_precision=dict()):
        assert not sample_point.is_refined()
        # Create parameter valuation
        storm_valuation = {self._parameters[p]: sp.RationalRF(val) for p, val in sample_point.get_valuation().items()}

        # Instantiate parametric DFT
        sample_dft = self._inst_checker_approx.instantiate(storm_valuation)
        sp.dft.compute_dependency_conflicts(sample_dft, use_smt=False, solver_timeout=0)  # Needed as instantiation looses this information

        if precision == 0:
            # Build complete CTMC from DFT
            print(' - Start building complete state space')
            if self._all_relevant:
                empty_sym = sp.dft.DFTSymmetries()
                relevant_events = sp.dft.compute_relevant_events(self._dft, [], ["all"])
                self._model = sp.dft.build_model(sample_dft, empty_sym, relevant_events)
            else:
                self._model = sp.dft.build_model(sample_dft, sample_dft.symmetries())
            self._init_state = self._model.initial_states[0]
            print(' - Finished building complete model')

            if self._model.model_type == sp.ModelType.MA:
                print("ERROR: Resulting model is MA instead of CTMC")
                assert False

            results = []
            for prop in self._properties:
                # Check CTMC
                results.append(sp.model_checking(self._model, prop).at(self._init_state))
            sample_point.set_results(results, refined=True)
            return sample_point

        # Compute approximation from DFT
        # Get existing builder
        assert sample_point in self._builders
        builder, iteration = self._builders[sample_point]
        print("  - Use existing builder with iteration {}".format(iteration))

        # Segfault occurs when trying to reuse builder
        # -> temporary solution is to rebuild to given iteration without model checking inbetween
        if self._all_relevant:
            empty_sym = sp.dft.DFTSymmetries()
            builder = stormpy.dft.ExplicitDFTModelBuilder_double(sample_dft, empty_sym)
        else:
            builder = stormpy.dft.ExplicitDFTModelBuilder_double(sample_dft, sample_dft.symmetries())
        for i in range(iteration + 1):
            builder.build(i, 1.0, sp.dft.ApproximationHeuristic.PROBABILITY)

        iteration += 1  # Next iteration
        print(' - Refine partial state space from iteration {}'.format(iteration))
        iterating = True
        while iterating:
            self._model = builder.get_partial_model(True, False)
            builder.build(iteration, 1.0, sp.dft.ApproximationHeuristic.PROBABILITY)
            self._model = builder.get_partial_model(True, False)
            self._init_state = self._model.initial_states[0]
            if self._model.model_type == sp.ModelType.MA:
                print("ERROR: Resulting model is MA instead of CTMC")
                assert False

            model_up = builder.get_partial_model(False, False)
            init_up = model_up.initial_states[0]
            results = []
            iterating = False

            # Try longest timebound first for faster abortion
            prop_last = self._properties[-1]
            result_last_low = stormpy.model_checking(self._model, prop_last).at(self._init_state)
            result_last_up = stormpy.model_checking(model_up, prop_last).at(init_up)
            print("   - Iteration {} ({} states): {}, {}".format(iteration, self._model.nr_states, result_last_low, result_last_up))
            if not slurf.util.is_precise_enough(result_last_low, result_last_up, precision, ind_precision, prop_last):
                iterating = True
                iteration += 1
                continue

            for prop in self._properties[:-1]:
                result_low = stormpy.model_checking(self._model, prop).at(self._init_state)
                result_up = stormpy.model_checking(model_up, prop).at(init_up)
                print("   - Iteration {}: {}, {}".format(iteration, result_low, result_up))
                if not slurf.util.is_precise_enough(result_low, result_up, precision, ind_precision, prop):
                    iterating = True
                    iteration += 1
                    break
                results.append((result_low, result_up))
            results.append((result_last_low, result_last_up))
        print(' - Finished building partial model after {} iterations'.format(iteration))
        # Store builder in cache
        self._builders[sample_point] = (builder, iteration)

        sample_point.set_results(results, refined=False)
        return sample_point


class DftSimulationSamplerInterface(DftModelSamplerInterface):
    """
    This simple interface builds a parametric DFT and simulates traces on the instantiated DFT.
    """

    def __init__(self, all_relevant=False, no_simulation=1000):
        super(DftModelSamplerInterface, self).__init__()
        self._simulation_results = dict()
        self._all_relevant = all_relevant
        self._no_simulations = no_simulation

    def load(self, model, properties, bisim=True, constants=None):
        """
        Note that load() only constructs the DFT. The simulation is performed for each sample individually.

        Parameters
        ----------
        :model: A DFT with parametric failure rates.
        :properties: Properties here is either a tuple (event, [time bounds]) or a list of properties.
        :bisim: Whether to apply bisimulation (not used for simulation)
        :constants: Constants for graph changing variables in model description (not required for fault trees)

        Returns Dictionary of parameters and their bounds.
        -------

        """
        time_start = time.process_time()
        print(' - Load DFT from Galileo file')
        # Load DFT from Galileo file
        self._dft = sp.dft.load_parametric_dft_galileo_file(model)
        # Make DFT well-formed
        self._dft = sp.dft.transform_dft(self._dft, unique_constant_be=True, binary_fdeps=True, exponential_distributions=True)
        # Check for dependency conflicts -> no conflicts mean CTMC
        sp.dft.compute_dependency_conflicts(self._dft, use_smt=False, solver_timeout=0)

        self._inst_checker = sp.dft.DFTInstantiator(self._dft)

        # Create properties
        self.prepare_properties(properties)
        # Obtain timebounds
        timebounds, target_label = get_timebounds_and_target(self._properties)
        assert target_label == "failed"
        self._timebounds = timebounds

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
        sample_dft = self._inst_checker.instantiate(storm_valuation)
        sp.dft.compute_dependency_conflicts(sample_dft, use_smt=False, solver_timeout=0)  # Needed as instantiation looses this information

        if self._all_relevant:
            empty_sym = sp.dft.DFTSymmetries()
            info = sample_dft.state_generation_info(empty_sym)
        else:
            info = sample_dft.state_generation_info(sample_dft.symmetries())

        seed = random.randrange(65535)
        generator = stormpy.dft.RandomGenerator.create(seed)
        simulator = stormpy.dft.DFTSimulator_double(sample_dft, info, generator)

        # Simulate
        print(' - Simulate DFT {} times'.format(self._no_simulations))
        results = []
        for timebound in self._timebounds:
            successful = 0
            for i in range(self._no_simulations):
                res = simulator.simulate_trace(timebound)
                if res == stormpy.dft.SimulationResult.SUCCESSFUL:
                    successful += 1
            results.append(successful)
        print(' - Finished simulation')
        self._simulation_results[sample_point] = (results, self._no_simulations)
        sample_point.set_results([x / self._no_simulations for x in results], refined=False)
        return sample_point

    def _refine(self, sample_point, precision, ind_precisions=dict()):
        assert not sample_point.is_refined()
        if sample_point in self._simulation_results:
            old_successful, no_sim = self._simulation_results[sample_point]
        else:
            old_successful, no_sim = [0] * len(self._timebounds), 0

        sample_point = self._sample(sample_point)
        new_successful, no_sim_additional = self._simulation_results[sample_point]
        total_successful = [sum(x) for x in zip(old_successful, new_successful)]
        total_simulation = no_sim + no_sim_additional
        self._simulation_results[sample_point] = (total_successful, total_simulation)

        sample_point.set_results([x / total_simulation for x in total_successful], refined=False)
        return sample_point

    def get_stats(self):
        return {
            "model_states": 0,
            "model_transitions": 0,
            "orig_model_states": 0,
            "orig_model_transitions": 0,
            "no_simulations": self._no_simulations,
            "no_parameters": len(self._parameters),
            "no_samples": len(self._samples.get_samples()),
            "no_properties": len(self._properties),
            "sample_calls": self._sample_calls,
            "refined_samples": self._refined_samples,
            "time_load": round(self._time_load, 4),
            "time_bisim": round(self._time_bisim, 4),
            "time_sample": round(self._time_sample, 4)
        }
