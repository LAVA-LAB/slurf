import stormpy as sp
import stormpy.pars
import stormpy.logic

import math


class ApproximationOptions:
    """
    Sets the hyperparameters used for approximation.
    """

    def __init__(self):
        self._fixed_states_absorbing = []
        self._max_depth_of_considered_states = 10

    def set_fixed_states_absorbing(self, ids):
        self._fixed_states_absorbing = ids

    @property
    def fixed_states_absorbing(self):
        return self._fixed_states_absorbing


class ApproximateChecker:
    """
    For a given pCTMC, check CTMC by approximating the CTMC.
    """

    def __init__(self, pctmc, options=ApproximationOptions()):
        self._original_model = pctmc
        self._abort_label = "deadl"
        self._subcheckers = dict()
        self._environment = sp.Environment()
        self._lb_formula = None
        self._ub_formula = None
        self._options = options

    def check(self, instantiation, inst_id):
        # TODO use an instantiation checker that yields transient probabilities
        checker, submodel, initial_state = self._get_submodel_instantiation_checker(instantiation, inst_id)
        checker.specify_formula(sp.ParametricCheckTask(self._lb_formula, True))  # Only initial states
        lb = checker.check(self._environment, instantiation).at(initial_state)
        if submodel.labeling.contains_label(self._abort_label):
            checker.specify_formula(sp.ParametricCheckTask(self._ub_formula, True))  # Only initial states
            ub = checker.check(self._environment, instantiation).at(initial_state)
        else:
            ub = lb
        return lb, ub

    def specify_formula(self, formula):
        self._lb_formula = formula
        # TODO once instantiation checker yields transient probabilities, this is no longer necessary
        assert type(formula.subformula.right_subformula) == sp.logic.AtomicLabelFormula
        old_label = formula.subformula.right_subformula.label
        self._ub_formula = stormpy.parse_properties(str(self._lb_formula).replace("\"" + old_label + "\"", "(\"" + old_label + "\" | \"" + self._abort_label + "\")"))[
            0].raw_formula

    def _get_submodel_instantiation_checker(self, instantiation, inst_id):
        if inst_id in self._subcheckers:
            # Use existing approximation for the same instantiation
            return self._subcheckers[inst_id]
        else:
            submodel = self._build_submodel(instantiation)
            assert len(submodel.initial_states) == 1
            init_state = submodel.initial_states[0]
            subchecker = sp.pars.PCtmcInstantiationChecker(submodel)
            self._subcheckers[inst_id] = (subchecker, submodel, init_state)
        return subchecker, submodel, init_state

    def _build_submodel(self, instantiation):
        selected_outgoing_transitions = self._select_states(instantiation)
        # Now build the submodel.
        options = sp.SubsystemBuilderOptions()
        options.fix_deadlocks = True
        submodel_result = sp.construct_submodel(self._original_model,
                                                sp.BitVector(self._original_model.nr_states, True),
                                                selected_outgoing_transitions, False, options)
        submodel = submodel_result.model
        assert submodel_result.deadlock_label is None or self._abort_label == submodel_result.deadlock_label
        return submodel

    def _select_states(self, instantiation):
        # TODO actually implement useful selection strategies here.
        selected_outgoing_transitions = sp.BitVector(self._original_model.nr_states, True)
        for id in self._options.fixed_states_absorbing:
            selected_outgoing_transitions.set(id, False)
        return selected_outgoing_transitions
