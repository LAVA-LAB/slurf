from slurf.model_sampler_interface import CtmcReliabilityModelSamplerInterface, DftReliabilityModelSamplerInterface
from . import util as testutils

import math


class TestModelSampler:
    def test_CTMC(self):
        sampler = CtmcReliabilityModelSamplerInterface()
        parameters_with_bounds = sampler.load(testutils.tiny_pctmc, ("full", [1, 5, 10]))
        parameter_objects_by_name = {p.name: p for p in parameters_with_bounds.keys()}
        assert "p" in parameter_objects_by_name
        result = sampler.sample(1, {parameter_objects_by_name["p"]: 0.3})
        assert math.isclose(result[0], 0.1734083474)
        assert math.isclose(result[1], 0.9427719189)
        assert math.isclose(result[2], 0.9987049333)

    # TODO: only one DFT call can be enabled because otherwise the setting '--exportexplicit' is set multiple times
    # def test_DFT(self):
    #     sampler = DftReliabilityModelSamplerInterface()
    #     parameters_with_bounds = sampler.load(testutils.dft_and, ("failed", [1, 5, 10]))
    #     parameter_objects_by_name = {p.name: p for p in parameters_with_bounds.keys()}
    #     assert "x" in parameter_objects_by_name
    #     result = sampler.sample(1, {parameter_objects_by_name["x"]: 0.5})
    #     assert math.isclose(result[0], 0.1548181217)
    #     assert math.isclose(result[1], 0.8425679498)
    #     assert math.isclose(result[2], 0.9865695059)

    def test_non_monotonic_DFT(self):
        sampler = DftReliabilityModelSamplerInterface()
        parameters_with_bounds = sampler.load(testutils.nonmonotonic_dft, ("failed", [0.1, 1, 2]))
        parameter_objects_by_name = {p.name: p for p in parameters_with_bounds.keys()}
        assert "x" in parameter_objects_by_name
        assert "y" in parameter_objects_by_name

        # First sample point
        point1 = {parameter_objects_by_name["x"]: 0.5, parameter_objects_by_name["y"]: 1}
        result = sampler.sample(1, point1)
        assert math.isclose(result[0], 0.692290873)
        assert math.isclose(result[1], 0.8773735196)
        assert math.isclose(result[2], 0.9548882389)

        # Second sample point
        point2 = {parameter_objects_by_name["x"]: 1, parameter_objects_by_name["y"]: 0.5}
        result = sampler.sample(1, point2)
        assert math.isclose(result[0], 0.524342103)
        assert math.isclose(result[1], 0.6967346701)
        assert math.isclose(result[2], 0.8160602794)
