import numpy as np
import cvxpy as cp
import math
import copy

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
        keep_mask = np.zeros(self.Nsamples, dtype=bool)

        # For every non-relaxed sample, check its dual values
        for i in range(self.Nsamples):
            LB_dual = sol['constraints_low'][i].dual_value
            UB_dual = sol['constraints_upp'][i].dual_value

            # If any xi is nonzero, sample definitely belongs to support set
            if any(~np.isclose(sol['xi'][i, :], 0)):
                keep_mask[i] = True

            else:
                # If all dual values are zero, sample is not associated with
                # any active constraint
                if all(np.isclose(LB_dual, 0)) and all(np.isclose(UB_dual, 0)):
                    support_mask[i] = False

        # Remove every sample which is definitely not of support
        samples_keep = self.samples[keep_mask]
        samples_left = self.samples[support_mask * ~keep_mask, :]

        print('Length keep set:', len(samples_keep))
        print('Length left set:', len(samples_left))
        print('Removed:', sum(support_mask == 0))

        # Iteratively remove other samples
        while True:

            if len(samples_left) == 0:
                break

            print('Samples left:',len(samples_left))
            n = 0

            mask = np.ones(len(samples_left), dtype=bool)
            mask[n] = False

            current_samples = np.vstack((samples_left[mask], samples_keep))

            _, solB = solve(current_samples, costOfRegret)

            # If the objective has changed, we must keep this sample
            if any(~np.isclose(sol['x_mean'], solB['x_mean'])) or \
               any(~np.isclose(sol['x_width'], solB['x_width'])):

                # Move current sample to the 'keep' set
                samples_keep = np.vstack((samples_keep, samples_left[n]))
                samples_left = samples_left[mask]

                print('Keep sample!')

            else:
            # If the objective is still the same, we can safely remove it

                # Remove current sample from set
                samples_left = samples_left[mask]

        complexity = len(samples_left) + len(samples_keep)
        print('Complexity is', complexity)

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
