import os
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from slurf.scenario_problem import scenarioProblem

from slurf.model_sampler_interface import \
    CtmcReliabilityModelSamplerInterface, \
    DftReliabilityModelSamplerInterface


def _path(folder, file):
    """
    Internal method for simpler listing of examples.
    :param folder: Folder.
    :param file: Example file.
    :return: Complete path to example file.
    """
    testfile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models")
    return os.path.join(testfile_dir, folder, file)


# Generate samples
Tend = 30
Tlist = np.arange(0, Tend+1, 5)

Nsamples = 100

MODEL = _path("", "sir.sm")

# Load model
sampler = CtmcReliabilityModelSamplerInterface()
parameters_with_bounds = sampler.load(MODEL, ("done", Tlist))
parameter_objects_by_name = {p.name: p for p in parameters_with_bounds.keys()}

assert "ki" in parameter_objects_by_name and "kr" in parameter_objects_by_name

samples = np.zeros((Nsamples, len(Tlist)))

fig, ax = plt.subplots()

for n in range(Nsamples):

    print('Start for sample', str(n) + '..')

    # Sample parameters
    ki, kr = np.random.uniform(low=[0.05, 0.05], high=[0.3, 0.3])

    # Set parameter values in dictionary
    param_values = {
        parameter_objects_by_name["ki"]: ki,
        parameter_objects_by_name["kr"]: kr
        }

    # Compute results
    samples[n] = list(sampler.sample(1, param_values).values())

    # if Nsamples < 1000 or n % int(Nsamples/100) == 0:
    plt.plot(Tlist, samples[n, :], color='k', lw=0.3, ls='dotted', alpha=0.3)

# Compute bounds on violation probability with risk and complexity theory

# Set colors and markers
colors = sns.color_palette()
markers = ['o', '*', 'x', '.', '+']

# Values of rho (cost of regret) at which to solve the scenario program
rho_list = [2, 1, 0.5, 0.3, 0.1]

x_low = {}
x_upp = {}
violation_prob = np.zeros(len(rho_list))

problem = scenarioProblem(samples)

for i, rho in enumerate(rho_list):

    x_mean, x_width, c_star, x_star = problem.rectangular(rho)

    print("\nThe optimal value is", x_star)
    print('Complexity of penalty-based program:', c_star)

    from compute_eta import etaLow, betaLow

    beta_desired = 0.99

    violation_prob[i] = 1 - etaLow(Nsamples, c_star, beta_desired)

    print('Upper bound on violation probability:', violation_prob[i])

    confidence_prob = betaLow(Nsamples, c_star, 1-violation_prob[i])

    print('Error with desired beta:', np.abs(beta_desired - confidence_prob))

    # Plot confidence regions
    labelStr = r'$\rho$: '+str(rho) + \
        '; $s^*$: '+str(c_star) + \
        '; $\epsilon$: ' + str(np.round(violation_prob[i], 2))

    x_low[i] = np.hstack((0, x_mean - x_width))
    x_upp[i] = np.hstack((0, x_mean + x_width))

    plt.plot(Tlist, x_low[i], lw=2, marker=markers[i],
             color=colors[i], label=labelStr)
    plt.plot(Tlist, x_upp[i], lw=2, marker=markers[i],
             color=colors[i])

plt.legend()
plt.show()
