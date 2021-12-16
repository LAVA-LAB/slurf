from slurf.approximate_ctmc_checker import ApproximateChecker
from . import util as testutils
import stormpy as sp
import pycarl as pc


class TestApproximateChecker:
    def test_tiny(self):
        program = sp.parse_prism_program(testutils.tiny_pctmc, True)
        properties = sp.parse_properties_for_prism_program("P=? [ F<=5 \"full\" ]", program)
        model = sp.build_parametric_model(program, properties)
        pars = model.collect_all_parameters()
        checker = ApproximateChecker(model, properties[0].raw_formula)
        instance = {p : pc.cln.Rational(0.5) for p in pars}
        lb, ub = checker.check(instance)
        assert lb <= 0.9112407511268089 <= ub


