import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
import seaborn as sns

from compute_eta import etaLow


class scenarioProblem:
    """
    Functions related to the scenario optimization part.
    """

    def __init__(self, samples):
        """

        Parameters
        ----------
        samples n x m array, with each row an m-dimensional sample

        """

        self.samples = samples
        self.Nsamples, self.dim = np.shape(self.samples)

    def rectangular(self, costOfRegret=1, debug=False):
        """

        Parameters
        ----------
        costOfRegret cost of violation parameter (rho in the paper)

        Returns
        -------
        Mean and width of optimal solution, complexity, and optimal cost.

        """

        # Solve initial problem with all constraints
        prob, sol = solve(self.samples, costOfRegret)

        x_star = prob.value

        support_mask = np.ones(self.Nsamples, dtype=bool)
        relaxed_mask = np.zeros(self.Nsamples, dtype=bool)

        # For every non-relaxed sample, check its dual values
        for i in range(self.Nsamples):
            LB_dual = sol['constraints_low'][i].dual_value
            UB_dual = sol['constraints_upp'][i].dual_value

            # If any xi is nonzero, sample definitely belongs to support set
            if any(~np.isclose(sol['xi'][i, :], 0)):
                relaxed_mask[i] = True

            else:
                # If all dual values are zero, sample is not associated with
                # any active constraint
                if all(np.isclose(LB_dual, 0)) and all(np.isclose(UB_dual, 0)):
                    support_mask[i] = False

        # Remove every sample which is definitely not of support
        samples_keep = self.samples[relaxed_mask]
        samples_left = self.samples[support_mask * ~relaxed_mask, :]

        nr_discarded = sum(relaxed_mask)

        print(' - Number of relaxed samples:', len(samples_keep))
        print(' - Number of remaining samples:', len(samples_left))
        print(' - Samples in interior of region:', sum(support_mask == 0))

        # Iteratively remove other samples
        while True:

            if len(samples_left) == 0:
                break

            mask = np.ones(len(samples_left), dtype=bool)
            mask[0] = False

            current_samples = np.vstack((samples_left[mask], samples_keep))

            _, solB = solve(current_samples, costOfRegret)

            # If the objective has changed, we must keep this sample
            if any(~np.isclose(sol['x_mean'], solB['x_mean'])) or \
               any(~np.isclose(sol['x_width'], solB['x_width'])):

                # Move current sample to the 'keep' set
                samples_keep = np.vstack((samples_keep, samples_left[0]))

            # Remove current sample from set
            samples_left = samples_left[mask]

        nr_support = len(samples_keep) - nr_discarded
        complexity = (nr_support, nr_discarded)

        return sol, complexity, x_star


def solve(samples, costOfRegret):

    Nsamples, dim = np.shape(samples)

    # Define convex optimization program
    x_mean = cp.Variable(dim, nonneg=True)
    x_width = cp.Variable(dim, nonneg=True)

    # Define regret/slack variables
    xi = cp.Variable((Nsamples, dim), nonneg=True)

    # Cost of violation
    rho = cp.Parameter()
    rho.value = costOfRegret

    constraints_low = []
    constraints_upp = []

    # Add constraints for each samples
    for n in range(Nsamples):

        constraints_low += [samples[n, :] >= x_mean - x_width - xi[n, :]]
        constraints_upp += [samples[n, :] <= x_mean + x_width + xi[n, :]]

    obj = cp.Minimize(sum(x_width) + rho * sum(xi @ np.ones(dim)))

    prob = cp.Problem(obj, constraints_low + constraints_upp)
    prob.solve() #solver='GUROBI')

    sol = {
        'x_mean': x_mean.value,
        'x_width': x_width.value,
        'xi': xi.value,
        'constraints_low': constraints_low,
        'constraints_upp': constraints_upp
        }

    return prob, sol


def compute_slurf(Tlist, samples, rho_list, beta=0.99):
    """

    Parameters
    ----------
    Tlist List of time points to evaluate at
    samples 2D Numpy array of samples
    Nsamples Number of samples
    rho_list List of the values of rho for which the scenario program is solved
    -------

    """

    Nsamples = len(samples)

    # Create plot
    fig, ax = plt.subplots()
    plt.plot(Tlist, samples.T, color='k', lw=0.3, ls='dotted', alpha=0.3)

    # Set colors and markers
    colors = sns.color_palette()
    markers = ['o', '*', 'x', '.', '+']

    x_low = {}
    x_upp = {}
    Pviolation = np.zeros(len(rho_list))

    problem = scenarioProblem(samples)

    for i, rho in enumerate(rho_list):

        print('\nInitialize scenario problem; cost of violation: {}'.format(rho))

        sol, c_star, x_star = problem.rectangular(rho)
        x_mean = sol['x_mean']
        x_width = sol['x_width']

        Pviolation[i] = np.round(1 - etaLow(Nsamples, sum(c_star), beta), 4)

        print('Upper bound on violation probability:', Pviolation[i])

        # Plot confidence regions
        labelStr = r'$\rho$: '+str(rho) + \
            r'; support: '+str(c_star[0]) + \
            r'; discard: '+str(c_star[1]) + \
            r'; $\epsilon$: ' + str(np.round(Pviolation[i], 2))

        x_low[i] = x_mean - x_width
        x_upp[i] = x_mean + x_width

        plt.plot(Tlist, x_low[i], lw=2, marker=markers[i],
                 color=colors[i], label=labelStr)
        plt.plot(Tlist, x_upp[i], lw=2, marker=markers[i],
                 color=colors[i])

    plt.xlabel('Time')
    plt.ylabel('Probability of zero infected')

    ax.set_title("Probability of zero infected population over time (N={} samples)".format(Nsamples))

    plt.legend(prop={'size': 6})
    plt.show()

    return Pviolation, x_low, x_upp
