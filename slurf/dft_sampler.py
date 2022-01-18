from slurf.ctmc_sampler import CtmcReliabilityModelSamplerInterface
from slurf.sample_cache import SampleCache

import stormpy as sp
import stormpy.pars
import stormpy.dft

import time


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

    def __init__(self):
        super(DftModelSamplerInterface, self).__init__()
        self._builders = dict()

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
        sp.dft.compute_dependency_conflicts(sample_dft, use_smt=False, solver_timeout=0)  # Needed as instantiation looses this information

        # Create new builder
        assert sample_point not in self._builders
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
            if not self.is_precise_enough(result_last_low, result_last_up, precision, ind_precision, prop_last):
                iterating = True
                iteration += 1
                continue

            for prop in self._properties[:-1]:
                result_low = stormpy.model_checking(self._model, prop).at(self._init_state)
                result_up = stormpy.model_checking(model_up, prop).at(init_up)
                print("   - Iteration {}: {}, {}".format(iteration, result_low, result_up))
                if not self.is_precise_enough(result_low, result_up, precision, ind_precision, prop):
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
