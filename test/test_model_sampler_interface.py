from slurf.model_sampler_interface import CtmcReliabilityModelSamplerInterface
from . import util as testutils


class TestModelSampler:
    def test_CTMC(self):
        sampler = CtmcReliabilityModelSamplerInterface()
        parameters_with_bounds = sampler.load(testutils.tiny_pctmc, ("full", [1,5,10]))
        parameter_objects_by_name = {p.name : p for p in parameters_with_bounds.keys()}
        sampler.sample(1, {parameter_objects_by_name["p"] : 0.3})



