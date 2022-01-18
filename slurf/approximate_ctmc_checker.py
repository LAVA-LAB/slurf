import stormpy as sp
import stormpy.pars
import stormpy.logic


class ApproximationOptions:
    """
    Sets the hyperparameters used for approximation.
    """

    def __init__(self, cluster_max_distance=1):
        self._max_depth_of_considered_states = 10
        self._cluster_max_distance = cluster_max_distance

    def set_cluster_max_distance(self, max_distance):
        self._cluster_max_distance = max_distance


class ApproximateChecker:
    """
    For a given pCTMC, check CTMC by approximating the CTMC.
    """

    def __init__(self, pctmc, model_desc=None, options=ApproximationOptions()):
        self._original_model = pctmc
        self._original_desc = model_desc
        self._abort_label = "deadl"
        self._subcheckers = dict()
        self._environment = sp.Environment()
        self._clusters = dict()
        self._options = options
        assert len(self._original_model.initial_states) == 1
        self._original_init_state = self._original_model.initial_states[0]
        self._inst_checker_exact = sp.pars.PCtmcInstantiationChecker(self._original_model)

    @staticmethod
    def formulas_lower_upper(formula, abort_label, model_desc=None):
        lb_formula = formula
        # TODO once instantiation checker yields transient probabilities, this is no longer necessary
        if type(formula.subformula.right_subformula) == sp.logic.AtomicLabelFormula:
            reach_label = "\"" + formula.subformula.right_subformula.label + "\""
        else:
            assert type(formula.subformula.right_subformula) == sp.logic.AtomicExpressionFormula
            assert model_desc is not None
            reach_label = str(formula.subformula.right_subformula.get_expression())
        ub_formula = ApproximateChecker.parse_property(str(lb_formula).replace(reach_label, "(" + reach_label + " | \"" + abort_label + "\")"), model_desc)
        return lb_formula, ub_formula, reach_label

    @staticmethod
    def parse_property(formula, model_desc=None):
        if model_desc is None:
            properties = sp.parse_properties(formula)
        else:
            if model_desc.is_prism_program:
                properties = sp.parse_properties_for_prism_program(formula, model_desc.as_prism_program())
            else:
                assert model_desc.is_jani_model
                properties = sp.parse_properties_for_jani_model(formula, model_desc.as_jani_model())
        return properties[0].raw_formula

    def check(self, sample_point, instantiation, formula):
        lb_formula, ub_formula, reach_label = ApproximateChecker.formulas_lower_upper(formula, self._abort_label, self._original_desc)

        # TODO use an instantiation checker that yields transient probabilities
        checker, submodel, initial_state = self._get_submodel_instantiation_checker(sample_point, instantiation, reach_label)

        # Check lower bound
        checker.specify_formula(sp.ParametricCheckTask(lb_formula, True))  # Only initial states
        lb = checker.check(self._environment, instantiation).at(initial_state)
        # print("Result for {}: {}".format(lb_formula, lb))

        # Check upper bound
        if submodel.labeling.contains_label(self._abort_label):
            checker.specify_formula(sp.ParametricCheckTask(ub_formula, True))  # Only initial states
            ub = checker.check(self._environment, instantiation).at(initial_state)
            # print("Result for {}: {}".format(ub_formula, ub))
        else:
            ub = lb
        return lb, ub

    def _get_submodel_instantiation_checker(self, sample_point, instantiation, reach_label):
        # TODO: handle refinement
        if sample_point.get_id() in self._subcheckers:
            # Use existing approximation for the same instantiation
            return self._subcheckers[sample_point.get_id()]

        # Find possible cluster
        cluster_point = self._find_nearby_cluster(sample_point)
        if cluster_point is not None:
            print("  - Use existing cluster")
            absorbing_states = self._clusters[cluster_point]
        else:
            # No cluster is close enough -> compute states to remove
            print("  - Use new cluster")
            absorbing_states = self._compute_absorbing_states(instantiation, reach_label)
            # Store cluster
            self._clusters[sample_point] = absorbing_states

        submodel = ApproximateChecker.build_submodel(self._original_model, absorbing_states, self._abort_label)
        assert len(submodel.initial_states) == 1
        init_state = submodel.initial_states[0]
        subchecker = sp.pars.PCtmcInstantiationChecker(submodel)
        self._subcheckers[sample_point.get_id()] = (subchecker, submodel, init_state)
        return subchecker, submodel, init_state

    @staticmethod
    def build_submodel(model, absorbing_states, abort_label):
        # Set outgoing transition to keep
        selected_outgoing_transitions = sp.BitVector(model.nr_states, True)
        for id in absorbing_states:
            selected_outgoing_transitions.set(id, False)

        # Now build the submodel
        options = sp.SubsystemBuilderOptions()
        options.fix_deadlocks = True
        submodel_result = sp.construct_submodel(model, sp.BitVector(model.nr_states, True), selected_outgoing_transitions, False, options)
        submodel = submodel_result.model
        sp.export_to_drn(submodel, "test_ctmc.drn")
        assert submodel_result.deadlock_label is None or abort_label == submodel_result.deadlock_label
        return submodel

    def _find_nearby_cluster(self, sample_point):
        best_point, best_distance = None, None
        for point in self._clusters.keys():
            distance = sample_point.get_distance(point)
            if distance <= self._options._cluster_max_distance:
                # Found nearby cluster
                if best_point is None or best_distance > distance:
                    best_point, best_distance = point, distance
        return best_point

    def _compute_absorbing_states(self, storm_valuation, reach_label):
        # Check expected time on CTMC to as heuristic for important/unimportant states
        formula = ApproximateChecker.parse_property(f'T=? [F {reach_label}]', self._original_desc)

        # Check CTMC
        self._inst_checker_exact.specify_formula(sp.ParametricCheckTask(formula, False))  # Get result for all states
        env = sp.Environment()
        result = self._inst_checker_exact.check(env, storm_valuation)
        assert result.result_for_all_states

        # Compute absorbing states
        result = dict(zip(range(self._original_model.nr_states), result.get_values()))
        res_reference = result[self._original_init_state] / 2.0
        return [i for i, res in result.items() if res < res_reference]
