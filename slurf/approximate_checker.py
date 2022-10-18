import slurf.util

import stormpy as sp
import stormpy.pars
import stormpy.logic

from queue import Queue
import math
from enum import Enum


class ApproxHeuristic(Enum):
    EXPECTED_TIME = 1
    REACH_PROB = 2


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

    def __init__(self):
        self._cluster_max_distance = 0
        self._heuristic = ApproxHeuristic.EXPECTED_TIME

    def set_cluster_max_distance(self, max_distance):
        self._cluster_max_distance = max_distance

    def set_approx_heuristic(self, heuristic):
        self._heuristic = heuristic


class SubModelInfo:
    """
    Stores relevant information for each submodel.
    """

    def __init__(self):
        self._model = None
        self._init_state = None
        self._instantiator = None
        self._absorbing_states = None
        self._iteration = 0
        self._exact = False
        self._state_expected_times = []
        self._state_reach_probabilities = []

    def set_absorbing_states(self, absorbing_states):
        self._absorbing_states = absorbing_states

    def get_absorbing_states(self):
        return self._absorbing_states

    def update_model(self, submodel):
        self._model = submodel
        assert len(self._model.initial_states) == 1
        self._init_state = self._model.initial_states[0]
        # Create instantiation model checkers
        if self._model.model_type == stormpy.ModelType.DTMC:
            self._instantiator = sp.pars.PDtmcInstantiator(self._model)
        elif self._model.model_type == stormpy.ModelType.CTMC:
            self._instantiator = sp.pars.PCtmcInstantiator(self._model)
        else:
            raise NotImplementedError("Model type {} not supported".format(self._model.model_type))


class ApproximateChecker:
    """
    For a given pDTMC or pCTMC, check the DTMC/CTMC by building a partial state space.
    """

    def __init__(self, pmc, model_desc=None, options=ApproximationOptions()):
        self._original_model = pmc
        self._original_desc = model_desc
        self._options = options
        assert len(self._original_model.initial_states) == 1
        self._original_init_state = self._original_model.initial_states[0]
        self._environment = sp.Environment()
        self._clusters = dict()
        self._abort_label = "deadl"
        self._formulas = []
        self._reach_label = None
        # Create instantiation model checkers
        if self._original_model.model_type == stormpy.ModelType.DTMC:
            self._instantiator_original = sp.pars.PDtmcInstantiator(self._original_model)
        elif self._original_model.model_type == stormpy.ModelType.CTMC:
            self._instantiator_original = sp.pars.PCtmcInstantiator(self._original_model)
        else:
            raise NotImplementedError("Model type {} not supported".format(self._model.model_type))

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
            print("   - Use existing cluster")
            submodel_info = self._clusters[cluster_point]
        else:
            # No cluster is close enough -> create new submodel info
            print("   - Use new cluster")
            cluster_point = sample_point
            submodel_info = SubModelInfo()
            submodel_info = self._compute_initial_absorbing_states(submodel_info, instantiation, self._reach_label, self._options._heuristic)
            submodel_info = ApproximateChecker.build_submodel(submodel_info, self._original_model, self._abort_label)

        # Instantiate parametric model
        inst_mc = submodel_info._instantiator.instantiate(instantiation)
        assert len(inst_mc.initial_states) == 1
        init_state = inst_mc.initial_states[0]

        while True:
            if submodel_info._exact:
                # Compute results on exact model
                results = []
                for lb_formula, _ in self._formulas:
                    results.append(self._compute_formula(inst_mc, lb_formula, init_state, self._environment))
                break
            else:
                results = self.compute_bounds(inst_mc, self._formulas, init_state, precision)
                if results is None:
                    # Refine further
                    submodel_info = self._refine_absorbing_states(submodel_info, instantiation, self._reach_label, self._options._heuristic)
                else:
                    # Precise enough
                    break
        print("   - Use model with {}/{} states".format(submodel_info._model.nr_states, self._original_model.nr_states))
        # Store cluster
        self._clusters[cluster_point] = submodel_info

        return results, submodel_info._exact

    def compute_bounds(self, model, formulas, init_state, precision):
        results = []
        for lb_formula, ub_formula in formulas:
            # Check lower bound
            lb = ApproximateChecker._compute_formula(model, lb_formula, init_state, self._environment)
            # Check upper bound
            if model.labeling.contains_label(self._abort_label):
                ub = ApproximateChecker._compute_formula(model, ub_formula, init_state, self._environment)
                assert slurf.util.leq(lb, ub)
                if not slurf.util.is_precise_enough(lb, ub, precision, dict(), None):
                    return None
            else:
                ub = lb
            results.append((lb, ub))
        return results

    @staticmethod
    def _compute_formula(model, formula, initial_state, environment):
        task = sp.CheckTask(formula, True)
        return sp.core._model_checking_sparse_engine(model, task, environment=environment).at(initial_state)

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
    def build_submodel(submodel_info, orig_model, abort_label):
        # Set outgoing transition to keep
        selected_outgoing_transitions = sp.BitVector(orig_model.nr_states, True)
        for id in submodel_info.get_absorbing_states():
            selected_outgoing_transitions.set(id, False)

        # Now build the submodel
        options = sp.SubsystemBuilderOptions()
        options.fix_deadlocks = True
        submodel_result = sp.construct_submodel(orig_model, sp.BitVector(orig_model.nr_states, True), selected_outgoing_transitions, False, options)
        submodel_info.update_model(submodel_result.model)
        # sp.export_to_drn(submodel, "test_mc.drn")
        assert submodel_result.deadlock_label is None or abort_label == submodel_result.deadlock_label
        if len(submodel_info.get_absorbing_states()) == 0:
            submodel_info._exact = True
        return submodel_info

    def _compute_initial_absorbing_states(self, submodel_info, instantiation, reach_label, heuristic):
        if heuristic == ApproxHeuristic.EXPECTED_TIME:
            return self._absorbing_states_by_expected_time(submodel_info, instantiation, reach_label, 1)
        elif heuristic == ApproxHeuristic.REACH_PROB:
            return self._absorbing_states_by_reachability_probability(submodel_info, instantiation, 0.05)
        else:
            print("ERROR: heuristic not known")
            assert False

    def _refine_absorbing_states(self, submodel_info, instantiation, reach_label, heuristic):
        # Get absorbing states
        if heuristic == ApproxHeuristic.EXPECTED_TIME:
            submodel_info = self._absorbing_states_by_expected_time(submodel_info, instantiation, reach_label, submodel_info._iteration + 1)
        elif heuristic == ApproxHeuristic.REACH_PROB:
            submodel_info = self._absorbing_states_by_reachability_probability(submodel_info, instantiation, 0.05 * math.pow(2, -submodel_info._iteration))
        else:
            print("ERROR: heuristic not known")
            assert False

        # Create new submodel
        submodel_info = ApproximateChecker.build_submodel(submodel_info, self._original_model, self._abort_label)
        submodel_info._iteration += 1
        return submodel_info

    def _absorbing_states_by_expected_time(self, submodel_info, instantiation, reach_label, threshold):
        # Check expected steps/time on DTMC/CTMC as heuristic for important/unimportant states
        if len(submodel_info._state_expected_times) == 0:
            # Initially compute expected times
            formula = ApproximateChecker.parse_property(f'T=? [F {reach_label}]', self._original_desc)

            # Check Markov chain
            inst_model = self._instantiator_original.instantiate(instantiation)
            task = sp.CheckTask(formula, False)  # Compute results for all states
            result = sp.core._model_checking_sparse_engine(inst_model, task, self._environment)
            assert result.result_for_all_states
            submodel_info._state_expected_times = result.get_values()

        # Compute absorbing states
        res_reference = submodel_info._state_expected_times[self._original_init_state] / (2 * threshold)
        submodel_info.set_absorbing_states([i for i, res in enumerate(submodel_info._state_expected_times) if res < res_reference])
        return submodel_info

    def _absorbing_states_by_reachability_probability(self, submodel_info, instantiation, threshold):
        # Compute reachability probability for each state as heuristic
        if len(submodel_info._state_reach_probabilities) == 0:
            # Initialize reachability probabilities by graph search
            inst_model = self._instantiator_original.instantiate(instantiation)

            submodel_info._state_reach_probabilities = [0] * inst_model.nr_states
            visited = sp.storage.BitVector(inst_model.nr_states)
            queue = Queue()
            # Start with initial state
            assert len(inst_model.initial_states) == 1
            init_state = inst_model.initial_states[0]
            queue.put(init_state)
            # Initial state is reachable with probability 1
            submodel_info._state_reach_probabilities[init_state] = 1
            while not queue.empty():
                current_state = queue.get()
                if visited.get(current_state):
                    continue
                visited.set(current_state, True)
                current_prob = submodel_info._state_reach_probabilities[current_state]
                #  Iterate through successor states
                s = inst_model.states[current_state]
                actions = s.actions
                assert len(actions) == 1
                action = actions[0]
                # Need two iterations: first compute exit rate
                exit_rate = 0
                for transition in action.transitions:
                    exit_rate += transition.value()
                # Then perform regular graph traversal
                for transition in action.transitions:
                    successor = transition.column
                    if successor == current_state:
                        # Ignore self-loops
                        continue
                    prob = transition.value() / exit_rate
                    # Update reachability probability
                    submodel_info._state_reach_probabilities[successor] += prob * current_prob
                    # Add state to queue if not visited
                    if not visited.get(successor):
                        queue.put(successor)

        for val in submodel_info._state_reach_probabilities:
            assert slurf.util.leq(val, 1)
        submodel_info.set_absorbing_states([i for i, prob in enumerate(submodel_info._state_reach_probabilities) if prob < threshold])
        return submodel_info
