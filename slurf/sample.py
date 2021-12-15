class Sample:

    def __init__(self, sample_id, valuation):
        self._id = sample_id
        self._result = []
        self.refined = False
        self._valuation = valuation

    def get_id(self):
        return self._id

    def get_valuation(self):
        return self._valuation

    def add_result(self, result):
        self._result.append(result)

    def get_result(self):
        return self._result

    def __str__(self):
        if self.refined:
            results = self.get_result()
        else:
            results = ["[{}, {}]".format(res[0], res[1]) for res in self.get_result()]
        return ", ".join(results)
