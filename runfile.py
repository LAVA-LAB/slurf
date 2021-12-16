import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import seaborn as sns
from slurf.scenario_problem import scenarioProblem

from slurf.model_sampler_interface import \
    CtmcReliabilityModelSamplerInterface


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


def sample_solutions(Nsamples, Tlist, model):
    """

    Parameters
    ----------
    Nsamples Number of samples
    Tlist List of time points to evaluate at
    model File name of the model to load

    Returns 2D Numpy array with every row being a sample
    -------

    """

    # Load model
    sampler = CtmcReliabilityModelSamplerInterface()
    parameters_with_bounds = sampler.load(model, ("done", Tlist))
    param_obj_by_name = {p.name: p for p in parameters_with_bounds.keys()}

    samples = np.zeros((Nsamples, len(Tlist)))
    parameters = np.random.uniform(low=[0.05, 0.05], high=[0.1, 0.1],
                                   size=(Nsamples, 2))

    for n in range(Nsamples):

        print('Start for sample', str(n) + '..')

        # Set parameters
        ki, kr = parameters[n]

        # Set parameter values in dictionary
        param_values = {
            param_obj_by_name["ki"]: ki,
            param_obj_by_name["kr"]: kr
            }

        # Compute results
        samples[n] = list(sampler.sample(1, param_values).values())

    return samples


def compute_slurf(Tlist, samples, Nsamples, rho_list):
    """

    Parameters
    ----------
    Tlist List of time points to evaluate at
    samples 2D Numpy array of samples
    Nsamples Number of samples
    rho_list List of the values of rho for which the scenario program is solved
    -------

    """

    # Create plot
    fig, ax = plt.subplots()
    plt.plot(Tlist, samples.T, color='k', lw=0.3, ls='dotted', alpha=0.3)

    # Set colors and markers
    colors = sns.color_palette()
    markers = ['o', '*', 'x', '.', '+']

    x_low = {}
    x_upp = {}
    violation_prob = np.zeros(len(rho_list))

    problem = scenarioProblem(samples)

    for i, rho in enumerate(rho_list):

        sol, c_star, x_star = problem.rectangular(rho)
        x_mean = sol['x_mean']
        x_width = sol['x_width']

        print("\nThe optimal value is", x_star)
        print('Complexity of penalty-based program:', c_star)

        from compute_eta import etaLow, betaLow

        beta_desired = 0.99

        violation_prob[i] = 1 - etaLow(Nsamples, c_star, beta_desired)

        print('Upper bound on violation probability:', violation_prob[i])

        confidence_prob = betaLow(Nsamples, c_star, 1-violation_prob[i])

        print('Error with desired beta:',
              np.abs(beta_desired - confidence_prob))

        # Plot confidence regions
        labelStr = r'$\rho$: '+str(rho) + \
            r'; $s^*$: '+str(c_star) + \
            r'; $\epsilon$: ' + str(np.round(violation_prob[i], 2))

        x_low[i] = x_mean - x_width
        x_upp[i] = x_mean + x_width

        plt.plot(Tlist, x_low[i], lw=2, marker=markers[i],
                 color=colors[i], label=labelStr)
        plt.plot(Tlist, x_upp[i], lw=2, marker=markers[i],
                 color=colors[i])

    plt.xlabel('Time')
    plt.ylabel('Probability of zero infected')

    ax.set_title("Probability of zero infected population over time (N={} samples)".format(Nsamples))

    plt.legend()
    plt.show()

    return x_low, x_upp


def printTime():

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    return current_time

# %%


print("Script started at:", printTime())

# Generate samples
Tlist = np.arange(10, 100+1, 15)

Nsamples = 100

# Generate given number of solutions to the parametric model
model = _path("", "sir.sm")
samples = sample_solutions(Nsamples, Tlist, model)

# Values of rho (cost of regret) at which to solve the scenario program
f = 8
rho_list = [10, 10/(f**1), 10/(f**2), 10/(f**3), 10/(f**4)]

# %%

# Compute SLURF and plot
x_low, x_upp = compute_slurf(Tlist, samples, Nsamples, rho_list)

print("Script done at:", printTime())
