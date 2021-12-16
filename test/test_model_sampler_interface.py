from slurf.model_sampler_interface import CtmcReliabilityModelSamplerInterface, DftReliabilityModelSamplerInterface
from . import util as testutils

import math


class TestModelSampler:

    def test_ctmc_refine(self):
        sampler = CtmcReliabilityModelSamplerInterface()
        parameters_with_bounds = sampler.load(testutils.tiny_pctmc, ("full", [1, 5, 10]))
        assert "p" in parameters_with_bounds
        sample = sampler.sample({"p": 0.3})
        result = sample.get_result()
        assert result[0][0] <= 0.1734083474 <= result[0][1]
        assert result[1][0] <= 0.9427719189 <= result[1][1]
        assert result[2][0] <= 0.9987049333 <= result[2][1]

        sample = sampler.refine(sample.get_id())
        result = sample.get_result()
        assert math.isclose(result[0], 0.1734083474)
        assert math.isclose(result[1], 0.9427719189)
        assert math.isclose(result[2], 0.9987049333)

    def test_ctmc_properties(self):
        sampler = CtmcReliabilityModelSamplerInterface()
        properties = ['P=? [ F<=5 "full" ]', 'P=? [ F=1 "empty" ]']
        parameters_with_bounds = sampler.load(testutils.tiny_pctmc, properties)
        assert "p" in parameters_with_bounds
        sample = sampler.sample({"p": 0.3}, exact=True)
        result = sample.get_result()
        assert math.isclose(result[0], 0.9427719189)
        assert math.isclose(result[1], 0.2720439223)

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

    def test_non_monotonic_dft(self):
        sampler = DftReliabilityModelSamplerInterface()
        parameters_with_bounds = sampler.load(testutils.nonmonotonic_dft, ("failed", [0.1, 1, 2]))
        assert "x" in parameters_with_bounds
        assert "y" in parameters_with_bounds

        # First sample point
        sample1 = sampler.sample({"x": 0.5, "y": 1}, exact=True)
        result1 = sample1.get_result()
        assert math.isclose(result1[0], 0.692290873)
        assert math.isclose(result1[1], 0.8773735196)
        assert math.isclose(result1[2], 0.9548882389)

        # Second sample point
        sample2 = sampler.sample({"x": 1, "y": 0.5}, exact=True)
        result2 = sample2.get_result()
        assert math.isclose(result2[0], 0.524342103)
        assert math.isclose(result2[1], 0.6967346701)
        assert math.isclose(result2[2], 0.8160602794)

    def test_batch_ctmc(self):
        sampler = CtmcReliabilityModelSamplerInterface()
        parameters_with_bounds = sampler.load(testutils.tiny_pctmc, ("full", [1, 5, 10]))
        assert "p" in parameters_with_bounds
        samples = sampler.sample_batch([{"p": 0.3}, {"p": 0.5}, {"p": 0.7}], exact=True)
        assert len(samples) == 3
        result0 = samples[0].get_result()
        assert math.isclose(result0[0], 0.1734083474)
        assert math.isclose(result0[1], 0.9427719189)
        assert math.isclose(result0[2], 0.9987049333)
        result1 = samples[1].get_result()
        assert math.isclose(result1[0], 0.1626966733)
        assert math.isclose(result1[1], 0.9112407511)
        assert math.isclose(result1[2], 0.9958568149)
        result2 = samples[2].get_result()
        assert math.isclose(result2[0], 0.1527927762)
        assert math.isclose(result2[1], 0.8761070032)
        assert math.isclose(result2[2], 0.9904796978)

    def test_ctmc_stats(self):
        sampler = CtmcReliabilityModelSamplerInterface()
        sampler.load(testutils.tiny_pctmc, ("full", [1, 5, 10]))
        sampler.sample({"p": 0.3})
        stats = sampler.get_stats()
        assert stats["model_states"] == 4
        assert stats['model_transitions:'] == 6
        assert stats['no_parameters'] == 1
        assert stats['no_samples'] == 1
        assert stats['no_properties'] == 3
        assert stats['sample_calls'] == 1
        assert stats['refined_samples'] == 0

    def test_refine_batch_ctmc(self):
        sampler = CtmcReliabilityModelSamplerInterface()
        parameters_with_bounds = sampler.load(testutils.tiny_pctmc, ("full", [1, 5, 10]))
        assert "p" in parameters_with_bounds
        samples = sampler.sample_batch([{"p": 0.3}, {"p": 0.5}, {"p": 0.7}])
        assert len(samples) == 3

        samples_refined = sampler.refine_batch([samples[0].get_id(), samples[2].get_id()])
        sample0 = samples_refined[0]
        assert sample0.is_refined()
        result0 = sample0.get_result()
        assert math.isclose(result0[0], 0.1734083474)
        assert math.isclose(result0[1], 0.9427719189)
        assert math.isclose(result0[2], 0.9987049333)
        sample2 = samples_refined[1]
        assert sample2.is_refined()
        result2 = sample2.get_result()
        assert math.isclose(result2[0], 0.1527927762)
        assert math.isclose(result2[1], 0.8761070032)
        assert math.isclose(result2[2], 0.9904796978)
