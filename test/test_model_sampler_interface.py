from slurf.model_sampler_interface import CtmcReliabilityModelSamplerInterface
from . import util as testutils

import math


class TestModelSampler:
    def test_CTMC(self):
        sampler = CtmcReliabilityModelSamplerInterface()
        parameters_with_bounds = sampler.load(testutils.tiny_pctmc, ("full", [1,5,10]))
        parameter_objects_by_name = {p.name : p for p in parameters_with_bounds.keys()}
        assert "p" in parameter_objects_by_name
        result = sampler.sample(1, {parameter_objects_by_name["p"] : 0.3})
        assert math.isclose(result[0], 0.1734083474)
        assert math.isclose(result[1], 0.9427719189)
        assert math.isclose(result[2], 0.9987049333)



