# -*- coding: utf-8 -*-
from slurf.commons import getDateTime
import pandas as pd

# Helper script to create an empty file for the table of benchmark statistics
EMPTY = pd.DataFrame(columns=['instance'])
EMPTY.set_index('instance', inplace=True)
file = 'Benchmark_statistics_='+str(getDateTime())
EMPTY.to_csv(file+'.csv')