from slurf.approximate_ctmc_checker import ApproximateChecker, ApproximationOptions
import slurf.util as util
from . import util as testutils
import stormpy as sp
import pycarl as pc


class TestApproximateChecker:
    def test_tiny(self):
        program = sp.parse_prism_program(testutils.mini_pctmc, True)
        properties = sp.parse_properties_for_prism_program("P=? [ F<=5 \"full\" ]", program)
        model = sp.build_parametric_model(program, properties)
        pars = model.collect_all_parameters()
        options = ApproximationOptions()
        options.set_fixed_states_absorbing([2])
        checker = ApproximateChecker(model, options)
        instance = {p: pc.cln.Rational(0.5) for p in pars}
        checker.specify_formula(properties[0].raw_formula, program)
        lb, ub = checker.check(instance, 0)
        assert util.is_inbetween(lb, 0.6996164004519213, ub)
