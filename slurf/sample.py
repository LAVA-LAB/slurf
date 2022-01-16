import math


class Sample:

    def __init__(self, sample_id, valuation):
        self._id = sample_id
        self._results = []
        self._refined = False
        self._valuation = valuation

    def get_id(self):
        return self._id

    def get_valuation(self):
        return self._valuation

    def get_distance(self, other):
        # Euclidean distance between two valuations
        point1 = []
        point2 = []
        for p, val in self.get_valuation().items():
            point1.append(val)
            point2.append(other.get_valuation()[p])
        return math.dist(point1, point2)

    def set_results(self, results, refined=False):
        self._results = results
        self._refined = refined

    def get_result(self):
        return self._results

    def is_refined(self):
        return self._refined

    def __str__(self):
        if self._refined:
            results = self.get_result()
        else:
            results = ["[{}, {}]".format(res[0], res[1]) for res in self.get_result()]
        return ", ".join(results)
