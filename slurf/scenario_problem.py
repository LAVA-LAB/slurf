import numpy as np
import cvxpy as cp

from slurf.compute_bound import etaLow


class scenarioProblem:
    """
    Functions related to the scenario optimization part.
    """

    def init_problem(self, samples, sample_ids):
        '''
        
        Parameters
        ----------
        samples Current subset of samples active in optimization problem
        sample_ids Indices of sample subset (compared to complete sample set)
        -------

        '''

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

    def solve_instance(self, disable_mask, costOfRegret, solver='ECOS'):

        # Set current parameters
        self.param.value = disable_mask
        self.rho.value = costOfRegret

        # Solve optimization problem
        if solver == 'ECOS':
            self.prob.solve(warm_start=True, solver='ECOS')
        elif solver == 'MOSEK':
            self.prob.solve(warm_start=True, solver='MOSEK')
        else:
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

    def solve(self, compareAt, costOfRegret=1):

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
            if all( (np.isclose(sol['xL'], solB['xL']))[compareAt] ) and \
               all( (np.isclose(sol['xU'], solB['xU']))[compareAt] ):

                support_mask[i] = 0

            del disable_mask

        complexity = sum(support_mask)

        print(' - Complexity:', complexity)

        # Compute IDs of the samples which are in the interior
        exterior_ids = self.sample_ids[~interior_mask]

        return sol, complexity, value, exterior_ids


def compute_solution_sets(samples, beta=0.99, rho_min=0.01, increment_factor=2,
                          itermax=8):
    """

    Parameters
    ----------
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

    regions = {}

    # Initialize scenario optimization problem
    problem = scenarioProblem()
    problem.init_problem(samples, np.arange(Nsamples))

    # Initialize value of rho (cost of violation)
    rho = rho_min
    i = 0
    exterior_ids = [None]
    
    tres = 1e-3
    compareAt = np.abs(np.max(samples, axis=0) - np.min(samples, axis=0)) > tres

    while len(exterior_ids) > 0 and i < itermax:

        print('\nSolve scenario problem of size {}; rho = {}'.\
              format(problem.Nsamples, rho))

        sol, c_star, x_star, exterior_ids = problem.solve(compareAt, rho)

        # If complexity is the same as in previous iteration, skip or break
        if i > 0 and c_star == regions[i-1]['complexity']:
            if c_star < 0.5*Nsamples:
                # If we are already at the outside of the slurf, break overall
                break
            else:
                # If we are still at the inside, increase rho and continue
                rho *= increment_factor
                continue
        
        # If complexity is equal to number of samples, the bound is not
        # informative (violation probability is one), so skip to next rho
        if c_star == Nsamples:
            rho *= increment_factor
            continue
        
        problem.init_problem(samples[exterior_ids], exterior_ids)

        Pviolation = np.round(1 - etaLow(Nsamples, c_star, beta), 4)

        print(' - Upper bound on violation probability:', Pviolation)

        regions[i] = {
            'x_low': sol['xL'],
            'x_upp': sol['xU'],
            'rho': rho,
            'complexity': c_star,
            'Pviolation': Pviolation
            }

        # Increment
        rho *= increment_factor
        i += 1

    return regions