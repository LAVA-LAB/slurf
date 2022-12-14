from slurf.sample import Sample

import pickle


class SampleCache:
    """
    Cache keeping track of all samples (valuations and their results).
    """

    def __init__(self):
        self._samples = list()
        self._max_id = 0

    def add_sample(self, valuation):
        sample_id = self._max_id
        assert len(self._samples) == sample_id
        self._max_id += 1
        sample = Sample(sample_id, valuation)
        self._samples.append(sample)
        
        self.num_samples = self._max_id
        
        return sample

    def get_sample(self, sample_id):
        assert sample_id < len(self._samples)
        return self._samples[sample_id]

    def get_samples(self):
        return self._samples


def export_sample_cache(sample_cache, file):
    samples_json = []
    for sample in sample_cache.get_samples():
        samples_json.append({
            "valuation": sample.get_valuation(),
            "result": sample.get_result(),
            "refined": sample.is_refined()
        })
    with open(file, 'wb') as out:
        pickle.dump(samples_json, out)


def import_sample_cache(file):
    with open(file, 'rb') as inp:
        samples_json = pickle.load(inp)
        samples = SampleCache()
        for sjson in samples_json:
            sample = samples.add_sample(sjson["valuation"])
            sample.set_results(sjson["result"], sjson["refined"])
        return samples
