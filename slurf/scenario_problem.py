import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import seaborn as sns

from slurf.compute_bound import etaLow


class scenarioProblem:
    """
    Functions related to the scenario optimization part.
    """

    def init_problem(self, samples, sample_ids):

        self.samples = samples
        self.sample_ids = sample_ids
        self.Nsamples, self.dim = np.shape(self.samples)

        # Define convex optimization program
        self.xU = cp.Variable(self.dim, nonneg=True)
        self.xL = cp.Variable(self.dim, nonneg=True)

        # Define regret/slack variables
        self.xi = cp.Variable((self.Nsamples, self.dim), nonneg=True)

        # Cost of violation
        self.rho = cp.Parameter()

        # Define slack variables (to enable/disable individual constraints)
        self.param = cp.Parameter(self.Nsamples)
        self.slack = cp.Variable(self.Nsamples)

        self.constraints_low = []
        self.constraints_upp = []

        # Add constraints
        for n in range(self.Nsamples):

            self.constraints_low += \
                [self.samples[n, :] >= self.xL -
                 self.xi[n, :] - cp.multiply(self.param[n], self.slack[n])]
            self.constraints_upp += \
                [self.samples[n, :] <= self.xU +
                 self.xi[n, :] + cp.multiply(self.param[n], self.slack[n])]

        # Objective function
        self.obj = cp.Minimize(sum(self.xU - self.xL) +
                               self.rho * cp.sum(self.xi))

        self.prob = cp.Problem(self.obj, [self.xU >= self.xL] +
                               self.constraints_low + self.constraints_upp)

    def solve_instance(self, disable_mask, costOfRegret):

        # Set current parameters
        self.param.value = disable_mask
        self.rho.value = costOfRegret

        # Solve optimization problem
        self.prob.solve(warm_start=True)

        # Return solution
        sol = {
            'xL': self.xL.value,
            'xU': self.xU.value,
            'xi': self.xi.value,
            'constraints_low': self.constraints_low,
            'constraints_upp': self.constraints_upp
            }

        return self.prob.value, sol

    def solve(self, costOfRegret=1):

        print('Solve problem of size:', self.Nsamples)

        # Solve initial problem with all constraints
        mask = np.zeros(self.Nsamples)
        value, sol = self.solve_instance(mask, costOfRegret)

        # Initialize masks
        interior_mask = np.zeros(self.Nsamples, dtype=bool)
        support_mask = np.ones(self.Nsamples, dtype=bool)

        # For every non-relaxed sample, check its dual values
        for i in range(self.Nsamples):
            LB_dual = sol['constraints_low'][i].dual_value
            UB_dual = sol['constraints_upp'][i].dual_value

            if all(np.isclose(LB_dual, 0)) and all(np.isclose(UB_dual, 0)):

                interior_mask[i] = True
                support_mask[i] = False

        print(' - Samples in interior of region:', sum(interior_mask))

        # Reduce the size of the problem by removing all samples in interior
        # self.init_problem(self.samples[~interior_mask])
        # support_mask = support_mask[~interior_mask]

        for i in range(self.Nsamples):

            # If current sample is in interior, or is relaxed, we can skip it
            if not support_mask[i] or any(sol['xi'][i] > 0.01):
                continue

            # Disable the samples that are: not of support and are not relaxed
            disable_mask = np.array(~support_mask, dtype=int)
            disable_mask[i] = 1

            # Solve the optimization problem
            _, solB = self.solve_instance(disable_mask, costOfRegret)

            # If the objective has not changed, this sample is not of support
            if all(np.isclose(sol['xL'], solB['xL'])) and \
               all(np.isclose(sol['xU'], solB['xU'])):

                support_mask[i] = 0

            del disable_mask

        complexity = sum(support_mask)

        print(' - Complexity:', complexity)

        # Compute IDs of the samples which are in the interior
        exterior_ids = self.sample_ids[~interior_mask]

        return sol, complexity, value, exterior_ids


def compute_slurf(Tlist, samples, beta=0.99, rho_min=0.01, factor=2,
                  itermax=8):
    """

    Parameters
    ----------
    Tlist List of time points to evaluate at
    samples 2D Numpy array of samples
    beta Confidence probability (close to one means good)
    rho_min Cost of violation used in first iteration
    factor Multiplication factor to increase rho with
    itermax Maximum number of iterations to do over increasing rho
    -------

    Returns
    ----------
    Pviolation, x_low, x_upp Results for the slurfs in all iterations
    ----------

    """

    Nsamples = len(samples)

    # Create plot
    fig, ax = plt.subplots()
    plt.plot(Tlist, samples.T, color='k', lw=0.3, ls='dotted', alpha=0.3)

    # Set colors and markers
    colors = sns.color_palette("Blues_r", n_colors=itermax)
    markers = ['o', '*', 'x', '.', '+', 'v', '1', 'p', 'X']

    x_low = {}
    x_upp = {}
    Pviolation = {}
    c_star = {}

    # Initialize scenario optimization problem
    problem = scenarioProblem()
    problem.init_problem(samples, np.arange(Nsamples))

    # Initialize value of rho (cost of violation)
    rho = rho_min
    i = 0
    exterior_ids = [None]

    while len(exterior_ids) > 0 and i < itermax:

        i_rel = i % len(markers)

        print('\n\nSolve scenario problem; rho = {}'.format(rho))

        sol, c_star[i], x_star, exterior_ids = problem.solve(rho)

        # If complexity is the same as in previous iteration, skip or break
        if i > 0 and c_star[i] == c_star[i-1]:
            if c_star[i] < 0.5*Nsamples:
                # If we are already at the outside of the slurf, break overall
                break
            else:
                # If we are still at the inside, increase rho and continue
                rho *= factor
                continue
        
        # If complexity is equal to number of samples, the bound is not
        # informative (violation probability is one), so skip to next rho
        if c_star[i] == Nsamples:
            rho *= factor
            continue

        x_low[i] = sol['xL']
        x_upp[i] = sol['xU']

        problem.init_problem(samples[exterior_ids], exterior_ids)

        print(' - Optimization problem solved')

        Pviolation[i] = np.round(1 - etaLow(Nsamples, c_star[i], beta), 4)

        print(' - Upper bound on violation probability:', Pviolation[i])

        # Plot confidence regions
        labelStr = r'$\rho$: '+str(np.round(rho, 2)) + \
            r'; complexity: '+str(c_star[i]) + \
            r'; $\epsilon$: ' + str(np.round(Pviolation[i], 2))

        plt.plot(Tlist, x_low[i], lw=0.5, marker=markers[i_rel],
                 color=colors[i_rel], label=labelStr)
        plt.plot(Tlist, x_upp[i], lw=0.5, marker=markers[i_rel],
                 color=colors[i_rel])

        # Increment
        rho *= factor
        i += 1

    plt.xlabel('Time')
    plt.ylabel('Probability of zero infected')

    ax.set_title("Probability of zero infected population (N={} samples)".
                 format(Nsamples))

    plt.legend(prop={'size': 6})
    plt.show()

    return Pviolation, x_low, x_upp
