import stormpy as sp
import stormpy.pars
import stormpy.dft
import math


class ModelSamplerInterface:
    """
    Describes the interface for sampling parametric models.
    """

    def __init__(self):
        self._model = None
        self._init_state = None
        self._properties = None
        self._parameters = None
        self._inst_checker = None

    def load(self, model, properties):
        """

        Parameters
        ----------
        model Description file for the (parametric) model
        properties An (ordered) set of properties to be evaluated

        Returns
        -------
        A dictionary that maps the parameter objects used for sampling later to upper and lower bounds.
        """

    def sample(self, id, sample_point, property_ids=None, store=False):
        """
        Analyse the model

        Parameters
        ----------
        id An identifier that can be used later if we want to get some refinement or the like
        sample_point The point that we want to sample in form of a dictionary from parameters to values
        property_ids The properties to be evaluated as indexed, if None, then all properties should be checked
        store Should we store the model to file for later reconsideration?

        Returns
        -------
        A dictionary with for every property id the result of evaluating the model on the corresponding property
        """

    def get_stats(self):
        """

        Returns
        -------
        Some object that can be exported to CSV or JSON with interesting stats
        """


class CtmcReliabilityModelSamplerInterface(ModelSamplerInterface):
    """
    This simple interface builds a parametric CTMC and then uses an instantiation checker to check the model.
    """

    def init_from_model(self, model):
        self._model = model
        # Get (unique) initial state
        assert len(self._model.initial_states) == 1
        self._init_state = self._model.initial_states[0]
        # Get parameters
        self._parameters = list(self._model.collect_all_parameters())
        # Create instantiation model checker
        self._inst_checker = sp.pars.PCtmcInstantiationChecker(self._model)
        # Return all parameters each with range (0 infinity)
        return {p: (0, math.inf) for p in self._parameters}

    def load(self, model, properties):
        """

        Parameters
        ----------
        model A CTMC with a label d
        properties Properties here is a tuple (event, [time bounds])

        Returns
        -------

        """
        # Load prism program
        program = sp.parse_prism_program(model)
        # Create properties
        event, time_bounds = properties[0], properties[1]
        property_strings = [f'P=? [ F<={float(t)} "{event}" ]' for t in time_bounds]
        self._properties = sp.parse_properties_for_prism_program(";".join(property_strings), program)
        # Build (sparse) CTMC
        options = sp.BuilderOptions([p.raw_formula for p in self._properties])
        model = sp.build_sparse_parametric_model_with_options(program, options)
        return self.init_from_model(model)

    def sample(self, id, sample_point, property_ids=None, store=False):
        # Set property ids (use all if none are given)
        prop_ids = range(len(self._properties)) if property_ids is None else property_ids

        # Check that sample point instantiates all parameters
        if len(sample_point) != len(self._parameters):
            print("ERROR: not all parameters are instantiated by sample point")
            assert False
        # Create sample point
        point = {p: sp.RationalRF(val) for p, val in sample_point.items()}
        # Sample point must be graph preserving
        self._inst_checker.set_graph_preserving(True)

        env = sp.Environment()
        results = {}
        # Analyse each property individually (Storm does not allow multiple properties for the InstantiationModelChecker
        for prop_id in prop_ids:
            # Specify formula
            formula = self._properties[prop_id].raw_formula
            self._inst_checker.specify_formula(sp.ParametricCheckTask(formula, True))  # Only initial states
            # Check CTMC
            result = self._inst_checker.check(env, point).at(self._init_state)
            # Add result
            results[prop_id] = result

        return results


class DftReliabilityModelSamplerInterface(CtmcReliabilityModelSamplerInterface):
    """
    This simple interface builds a parametric DFT, generates the corresponding parametric CTMC
    and then uses an instantiation checker to check the model.
    """

    def __init__(self):
        super(ModelSamplerInterface, self).__init__()
        self._dft = None

    def load(self, model, properties):
        """

        Parameters
        ----------
        model A DFT with parametric failure rates.
        properties Properties here is a tuple (event, [time bounds]).

        Returns Dictionary of parameters and their bounds.
        -------

        """
        # Load DFT from Galileo file
        dft = sp.dft.load_parametric_dft_galileo_file(model)

        # Create properties
        event, time_bounds = properties[0], properties[1]
        property_strings = [f'P=? [ F<={float(t)} "{event}" ]' for t in time_bounds]
        self._properties = sp.parse_properties(";".join(property_strings))

        # Use the first property to generate the CTMC state space
        # Obtain CTMC by exporting to DRN format and loading again
        # TODO: implement dedicated methods
        drn_file = "tmp_ctmc.drn"
        sp.set_settings(["--io:exportexplicit", drn_file])
        tmp_prop = sp.parse_properties(f'T=? [ F "{event}" ]')[0]
        stormpy.dft.analyze_parametric_dft(dft, [tmp_prop.raw_formula])
        return self.init_from_model(sp.build_parametric_model_from_drn(drn_file))
