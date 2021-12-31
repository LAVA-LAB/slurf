import numpy as np
from slurf.scenario_problem import compute_slurf

Tlist = np.arange(10, 140+1, 5)

results = np.genfromtxt(
    '/home/thom/Documents/slurf/input/reliability_curve_all25.csv', 
    delimiter=',', 
    skip_header=1)

results = 1 - results[:, 1:].T

min_rho = 0.03
increment_factor = 2
beta = 0.9

# Compute SLURF and plot
Pviolation, x_low, x_upp = compute_slurf(Tlist, results, beta,
                                         min_rho, increment_factor)