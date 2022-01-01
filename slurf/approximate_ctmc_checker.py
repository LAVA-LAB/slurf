import stormpy as sp
import stormpy.pars
import stormpy.logic


class ApproximationOptions:
    """
    Sets the hyperparameters used for approximation.
    """
    def __init__(self, max_depth_of_considered_states=10000000000):
        self._fixed_states_absorbing = []
        self._max_depth_of_considered_states = max_depth_of_considered_states

    def set_fixed_states_absorbing(self, ids):
        self._fixed_states_absorbing = ids

    @property
    def fixed_states_absorbing(self):
        return self._fixed_states_absorbing

    @property
    def max_depth_of_considered_states(self):
        return self._max_depth_of_considered_states


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
        self._target_label = None
        self._options = options

    class SubCheckerContainer:
        """
        Hold a subchecker and a pCTMC approximation
        """
        def __init__(self, checker, original_model, abort_label):
            self._checker = checker
            self._original_model = original_model
            self._abort_label = abort_label

        def get_bounds(self, environment, instantiation, lb_formula, ub_formula):
            # TODO use an instantiation checker that yields transient probabilities
            # Then, formulas no longer need to be passed
            self._checker.specify_formula(sp.ParametricCheckTask(lb_formula, True))  # Only initial states
            lb = self._checker.check(environment, instantiation).at(self._initial_state)
            if self._is_approximation:
                self._checker.specify_formula(sp.ParametricCheckTask(ub_formula, True))  # Only initial states
                ub = self._checker.check(environment, instantiation).at(self._initial_state)
            else:
                ub = lb
            return lb, ub

        @property
        def _is_approximation(self):
            return self._original_model.labeling.contains_label(self._abort_label)

        @property
        def _initial_state(self):
            return self._original_model.initial_states[0]

    def check(self, instantiation):
        checker = self._get_submodel_instantiation_checker(instantiation)
        return checker.get_bounds(self._environment, instantiation, self._lb_formula, self._ub_formula)

    def specify_formula(self, formula):
        self._lb_formula = formula
        # TODO once instantiation checker yields transient probabilities, this is no longer necessary
        # TODO this is a bit of a hack, but
        assert type(formula.subformula.right_subformula) == sp.logic.AtomicLabelFormula, "Currently only labels are allowed to specify target states"
        old_label = formula.subformula.right_subformula.label
        self._target_label = old_label
        self._ub_formula = stormpy.parse_properties(str(self._lb_formula).replace("\"" + old_label + "\"", "(\"" + old_label + "\" | \"" + self._abort_label + "\")"))[0].raw_formula

    def _get_submodel_instantiation_checker(self, instantiation):
        if "all" in self._subcheckers:
            # TODO Check whether we can use a previous approximation.
            result = self._subcheckers["all"]
        else:
            submodel = self._build_submodel(instantiation)
            assert len(submodel.initial_states) == 1
            subchecker = sp.pars.PCtmcInstantiationChecker(submodel)
            result = ApproximateChecker.SubCheckerContainer(subchecker, submodel, self._abort_label)
            self._subcheckers["all"] = result
        return result

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
        notavoidstates = sp.BitVector(self._original_model.nr_states, True) # TODO We here assume that there are no avoid states.
        target_states = self._original_model.labeling.get_states(self._target_label)
        reachable_states = sp.get_reachable_states(self._original_model, self._original_model.initial_states_as_bitvector,
                                 notavoidstates, target_states,
                                 maximal_steps=self._options._max_depth_of_considered_states)
        selected_outgoing_transitions = reachable_states
        for id in self._options.fixed_states_absorbing:
            selected_outgoing_transitions.set(id, False)
        return selected_outgoing_transitions
