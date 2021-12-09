import stormpy as sp
import math

class ModelSamplerInterface:
    """
    Describes the interface for sampling parametric models.
    """

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
    This simple interface builds a parametric CTMC and then uses an instantation checker to check the model.
    """

    def __init__(self):
        self._model = None
        self._properties = None
        self._parameters = None

    def load(self, model, properties):
        """

        Parameters
        ----------
        model A CTMC with a label d
        properties Properties here is a tuple (event, [time bounds])

        Returns
        -------

        """
        program = sp.parse_prism_program(model)
        event, time_bounds = properties[0], properties[1]
        property_strings = [f'P=? [ F<={float(t)} "{event}" ]' for t in time_bounds]
        self._properties = sp.parse_properties_for_prism_program(";".join(property_strings), program)
        options = sp.BuilderOptions([p.raw_formula for p in self._properties])
        self._model = sp.build_sparse_parametric_model_with_options(program, options)
        self._parameters = list(self._model.collect_all_parameters())
        # TODO instantiation checker
        return { p: (0, math.inf) for p in self._parameters}


    def sample(self, id, sample_point, property_ids=None, store=False):
        # TODO query instantiatino checker.
        pass


