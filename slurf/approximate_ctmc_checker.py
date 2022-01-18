import slurf.util

import stormpy as sp
import stormpy.pars
import stormpy.logic


def find_nearby_cluster(clusters, sample_point, max_distance):
    if sample_point in clusters:
        return sample_point
    best_point, best_distance = None, None
    for point in clusters.keys():
        distance = sample_point.get_distance(point)
        if distance <= max_distance:
            # Found nearby cluster
            if best_point is None or best_distance > distance:
                best_point, best_distance = point, distance
    return best_point


class ApproximationOptions:
    """
    Sets the hyperparameters used for approximation.
    """

    def __init__(self, cluster_max_distance=1):
        self._max_depth_of_considered_states = 10
        self._cluster_max_distance = cluster_max_distance

    def set_cluster_max_distance(self, max_distance):
        self._cluster_max_distance = max_distance


class SubModelInfo:
    """
    Stores relevant information for each submodel.
    """

    def __init__(self, submodel):
        self._model = submodel
        assert len(self._model.initial_states) == 1
        self._init_state = self._model.initial_states[0]
        self._inst_checker = sp.pars.PCtmcInstantiationChecker(self._model)
        self._absorbing_states = None
        self._iteration = 0
        self._exact = False


class ApproximateChecker:
    """
    For a given pCTMC, check CTMC by approximating the CTMC.
    """

    def __init__(self, pctmc, model_desc=None, options=ApproximationOptions()):
        self._original_model = pctmc
        self._original_desc = model_desc
        self._options = options
        assert len(self._original_model.initial_states) == 1
        self._original_init_state = self._original_model.initial_states[0]
        self._environment = sp.Environment()
        self._clusters = dict()
        self._inst_checker_exact = sp.pars.PCtmcInstantiationChecker(self._original_model)
        self._abort_label = "deadl"
        self._formulas = []
        self._reach_label = None

    def check(self, sample_point, instantiation, properties, precision, ind_precisions=dict()):
        if ind_precisions:
            print("ERROR: Individual precisions are currently not supported!")
            assert False

        if len(self._formulas) != len(properties):
            # Set new formulas
            self._formulas = []
            self._reach_label = None
            for prop in properties:
                lb_formula, ub_formula, r_label = ApproximateChecker.formulas_lower_upper(prop.raw_formula, self._abort_label, self._original_desc)
                self._formulas.append((lb_formula, ub_formula))
                if self._reach_label is None:
                    self._reach_label = r_label
                else:
                    assert self._reach_label == r_label

        # TODO use an instantiation checker that yields transient probabilities

        # Find possible cluster
        cluster_point = find_nearby_cluster(self._clusters, sample_point, self._options._cluster_max_distance)
        if cluster_point is not None:
            print("  - Use existing cluster")
            submodel_info = self._clusters[cluster_point]
        else:
            # No cluster is close enough -> create new submodel info
            print("  - Use new cluster")
            cluster_point = sample_point
            absorbing_states = self._compute_initial_absorbing_states(instantiation, self._reach_label)
            submodel_info = ApproximateChecker.build_submodel(self._original_model, absorbing_states, self._abort_label)

        while True:
            if submodel_info._exact:
                # Compute results on exact model
                results = []
                for lb_formula, _ in self._formulas:
                    results.append(self._compute_formula(submodel_info, instantiation, lb_formula))
                break
            else:
                print("Iteration start" + str(submodel_info._iteration))
                results = self.compute_bounds(submodel_info, instantiation, self._formulas, precision)
                if results is None:
                    # Refine further
                    submodel_info = self._refine_states(submodel_info, instantiation, self._reach_label)
                    print("Iteration refine" + str(submodel_info._iteration))
                else:
                    # Precise enough
                    break
        # Store cluster
        print("Iteration end" + str(submodel_info._iteration))

        self._clusters[cluster_point] = submodel_info
        if submodel_info._exact:
            print("Results for exact model: {}".format(results))
        else:
            print("Results for iteration {}: {}".format(submodel_info._iteration, results))

        return results, submodel_info._exact

    def compute_bounds(self, model_info, instantiation, formulas, precision):
        results = []
        for lb_formula, ub_formula in formulas:
            # Check lower bound
            lb = self._compute_formula(model_info, instantiation, lb_formula)
            # Check upper bound
            if model_info._model.labeling.contains_label(self._abort_label):
                ub = self._compute_formula(model_info, instantiation, ub_formula)
                assert slurf.util.leq(lb, ub)
                if not slurf.util.is_precise_enough(lb, ub, precision, dict(), None):
                    return None
            else:
                ub = lb
            results.append((lb, ub))
        return results

    def _compute_formula(self, model_info, instantiation, formula):
        model_info._inst_checker.specify_formula(sp.ParametricCheckTask(formula, True))  # Only initial states
        return model_info._inst_checker.check(self._environment, instantiation).at(model_info._init_state)

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
        # sp.export_to_drn(submodel, "test_ctmc.drn")
        assert submodel_result.deadlock_label is None or abort_label == submodel_result.deadlock_label
        return SubModelInfo(submodel)

    def _compute_initial_absorbing_states(self, instantiation, reach_label):
        # Check expected time on CTMC to as heuristic for important/unimportant states
        formula = ApproximateChecker.parse_property(f'T=? [F {reach_label}]', self._original_desc)

        # Check CTMC
        self._inst_checker_exact.specify_formula(sp.ParametricCheckTask(formula, False))  # Get result for all states
        env = sp.Environment()
        result = self._inst_checker_exact.check(env, instantiation)
        assert result.result_for_all_states

        # Compute absorbing states
        result = dict(zip(range(self._original_model.nr_states), result.get_values()))
        res_reference = result[self._original_init_state] / 2.0
        return [i for i, res in result.items() if res < res_reference]

    def _refine_states(self, submodel_info, instantiation, reach_label):
        # Check expected time on CTMC to as heuristic for important/unimportant states
        formula = ApproximateChecker.parse_property(f'T=? [F {reach_label}]', self._original_desc)

        # Check CTMC
        self._inst_checker_exact.specify_formula(sp.ParametricCheckTask(formula, False))  # Get result for all states
        env = sp.Environment()
        result = self._inst_checker_exact.check(env, instantiation)
        assert result.result_for_all_states

        # Compute absorbing states
        result = dict(zip(range(self._original_model.nr_states), result.get_values()))
        res_reference = result[self._original_init_state] / (2 * (submodel_info._iteration + 1))
        absorbing_states = [i for i, res in result.items() if res < res_reference]

        # Create new submodel
        new_info = ApproximateChecker.build_submodel(self._original_model, absorbing_states, self._abort_label)
        new_info._iteration = submodel_info._iteration + 1
        if not absorbing_states:
            new_info._exact = True
        return new_info
