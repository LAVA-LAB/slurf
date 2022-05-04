from slurf.model_sampler_interface import ModelSamplerInterface
from slurf.approximate_ctmc_checker import ApproximateChecker

import stormpy as sp
import stormpy.pars

import math
import os.path
import time


class CtmcReliabilityModelSamplerInterface(ModelSamplerInterface):
    """
    This simple interface builds a parametric CTMC and then uses an 
    instantiation checker to check the model.
    """

    def init_from_model(self, model, bisim=True):
        """
        Initialize sampler from CTMC model.

        Parameters
        ----------
        model CTMC.

        Returns Dict of all params. and their bounds (default [0, infinity)).
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
        self._instantiator = sp.pars.PCtmcInstantiator(self._model)
        self._inst_checker_approx = ApproximateChecker(self._model, self._symb_desc, self._approx_options)

        # Return all parameters each with range (0 infinity)
        return {p: (0, math.inf) for p in self._parameters.keys()}

    def prepare_properties(self, properties):
        """
        Set properties.

        Parameters
        ----------
        properties Properties either given as a tuple (event, [time bounds]) or
        a list of properties.
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
        properties Properties here is either a tuple (event, [time bounds]) or 
        a list of properties.
        bisim Whether to apply bisimulation.
        constants Constants for graph changing variables (optional)

        Returns Dict of all params. and their bounds (default [0, infinity)).
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

        # Check CTMC using approximation checker
        precision = 10
        results, exact = self._inst_checker_approx.check(sample_point, storm_valuation, self._properties, precision)
        # Add result
        sample_point.set_results(results, refined=exact)
        return sample_point

    def _refine(self, sample_point, precision, ind_precisions=dict()):
        assert not sample_point.is_refined()
        # Create parameter valuation
        storm_valuation = {self._parameters[p]: sp.RationalRF(val) for p, val in sample_point.get_valuation().items()}

        if precision == 0:
            # Compute exact results
            # Instantiate model
            inst_model = self._instantiator.instantiate(storm_valuation)

            env = sp.Environment()
            # Analyse each property individually (Storm does not allow multiple properties)
            results = []
            for prop in self._properties:
                # Check CTMC
                results.append(stormpy.model_checking(inst_model, prop).at(self._init_state))
            # Add result
            sample_point.set_results(results, refined=True)
            return sample_point

        # Compute precise enough results using approximation checker
        results, exact = self._inst_checker_approx.check(sample_point, storm_valuation, self._properties, precision, ind_precisions)
        # Add result
        sample_point.set_results(results, refined=exact)
        return sample_point
