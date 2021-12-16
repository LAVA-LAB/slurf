from slurf.sample_cache import SampleCache, import_sample_cache, export_sample_cache

import os


class TestSampleCache:
    def test_import_export(self, tmp_path):
        sample_cache = SampleCache()
        sample0 = sample_cache.add_sample({"p": 0.5})
        assert sample0._id == 0
        sample1 = sample_cache.add_sample({"p": 0.7})
        sample2 = sample_cache.add_sample({"p": 0.2})
        sample2.add_result(0.6)
        assert sample1._id == 1
        assert sample2._id == 2
        assert sample_cache.get_sample(2).get_valuation()["p"] == 0.2

        tmp_file = os.path.join(tmp_path, "samples.pkl")
        export_sample_cache(sample_cache, tmp_file)

        samples = import_sample_cache(tmp_file)
        assert samples.get_sample(2).get_valuation()["p"] == 0.2
        assert samples.get_sample(2).get_result()[0] == 0.6
