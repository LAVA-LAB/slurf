from slurf.approximate_ctmc_checker import ApproximateChecker
from slurf.sample_cache import SampleCache
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
        checker = ApproximateChecker(model, sp.SymbolicModelDescription(program))
        sample_cache = SampleCache()
        sample = sample_cache.add_sample({"p": 0.5})
        instance = {p: pc.cln.Rational(0.5) for p in pars}
        results, exact = checker.check(sample, instance, properties, precision=10)
        assert not exact
        assert util.is_inbetween(results[0][0], 0.6996164004519213, results[0][1])
