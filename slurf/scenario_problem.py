import numpy as np
import cvxpy as cp
import math

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

    def rectangular(self, costOfRegret = 1):
        """

        Parameters
        ----------
        costOfRegret cost of violation parameter (rho in the paper)

        Returns
        -------
        Mean and width of optimal solution, complexity, and optimal cost.

        """

        obj, x_mean, x_width, prob, constraints_low, constraints_upp = \
            self._solve(self.samples, costOfRegret)

        x_star = prob.value

        # Determine complexity of the solution
        support_set = np.ones(self.Nsamples)

        # For every non-relaxed sample, check its dual values
        for i in range(self.Nsamples):
            # # Otherwise, check its dual values

            LB_dual = constraints_low[i].dual_value
            UB_dual = constraints_upp[i].dual_value

            # If all dual values are zero, sample is not associated with any
            # active constraint
            active_count = sum(~np.isclose(LB_dual, 0)) + sum(~np.isclose(UB_dual, 0))
            support_set[i] = active_count

        # Remove every sample which is definitely not of support
        samples_bar = self.samples[support_set > 0, :]

        # Iteratively remove other samples
        done = False
        while not done:

            something_removed = False

            for n in range(len(samples_bar)):
                mask = np.zeros(len(samples_bar), dtype=bool)
                mask[n] = 1

                _, y_mean, y_width, _, _, _ = self._solve(samples_bar[mask], costOfRegret)

                # If objective is still the same, remove this sample
                if all(np.isclose(x_mean.value, y_mean.value)) and \
                   all(np.isclose(x_width.value, y_width.value)):
                    something_removed = True

                    # Remove current sample from set
                    samples_bar = samples_bar[mask]
                    break

            if not something_removed:
                done = True

        complexity = len(samples_bar)
        print('Complexity ')

        return x_mean.value, x_width.value, complexity, x_star

    def _solve(self, samples, costOfRegret):

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
        prob.solve(solver='GUROBI')

        return obj, x_mean, x_width, prob, constraints_low, constraints_upp
