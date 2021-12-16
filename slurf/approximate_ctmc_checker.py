import stormpy as sp
import stormpy.pars

import math


class ApproximateChecker:
    def __init__(self, pctmc):
        self._original_model = pctmc
        self._abort_label = "deadl"
        self._subcheckers = dict()
        self._environment = sp.Environment()
        self._lb_formula = None

    def check(self, instantiation):
        checker, initial_state = self._get_submodel_instantiation_checker(instantiation)
        checker.specify_formula(sp.ParametricCheckTask(self._lb_formula, True))  # Only initial states
        lb = checker.check(self._environment, instantiation).at(initial_state)
        # TODO specify correct formula
        ub = math.inf
        return lb, ub

    def specify_formula(self, formula):
        self._lb_formula = formula

    def _get_submodel_instantiation_checker(self, instantiation):
        if "all" in self._subcheckers:
            # TODO Check whether we can use a previous approximation.
            subchecker, init_state = self._subcheckers["all"]
        else:
            submodel = self._build_submodel(instantiation)
            assert len(submodel.initial_states) == 1
            init_state = submodel.initial_states[0]
            subchecker = sp.pars.PCtmcInstantiationChecker(submodel)
            self._subcheckers["all"] = (subchecker, init_state)
        return subchecker, init_state

    def _build_submodel(self, instantiation):
        selected_outgoing_transitions = self._select_states(instantiation)
        # Now build the submodel.
        options = sp.SubsystemBuilderOptions()
        options.fix_deadlocks = True
        submodel_result = sp.construct_submodel(self._original_model,
                                                sp.BitVector(self._original_model.nr_states, True),
                                                selected_outgoing_transitions, False, options)
        submodel = submodel_result.model
        assert self._abort_label == submodel_result.deadlock_label
        return submodel

    def _select_states(self, instantiation):
        # TODO actually implement useful selection strategies here.
        selected_outgoing_transitions = sp.BitVector(self._original_model.nr_states, True)
        selected_outgoing_transitions.set(3, False)
        selected_outgoing_transitions.set(1, False)
        return selected_outgoing_transitions
