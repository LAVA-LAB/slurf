import numpy as np
import cvxpy as cp
import pandas as pd

from slurf.compute_bound import etaLow


class scenarioProblem:
    """
    Functions related to the scenario optimization part.
    """

    def init_problem(self, samples, sample_ids, pareto_pieces=0):
        '''
        
        Parameters
        ----------
        samples Current subset of samples active in optimization problem
        sample_ids Indices of sample subset (compared to complete sample set)
        -------

        '''

        self.samples = samples
        self.sample_ids = sample_ids
        
        # Check if imprecise samples are used (means that dimension = 3)
        if self.samples.ndim == 3:
            imprecise = True
            self.Nsamples, self.dim, _ = np.shape(self.samples)
        else:
            imprecise = False
            self.Nsamples, self.dim = np.shape(self.samples)

        if pareto_pieces > 0: 
            if self.dim != 2:
                print('ERROR: Can currently only solve for Pareto-front in 2D')
                assert False
            
            self.paretoBase = cp.Variable(pareto_pieces, nonneg=True)
            self.paretoSlope  = cp.Variable(pareto_pieces, nonneg=True)
            self.pareto = True
        else:
            self.pareto = False

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

            # Switch between precise vs. imprecise samples
            if imprecise:
                
                # For imprecise samples, formulate constraint such that the
                # whole box sample is contained in the confidence region
                self.constraints_low += \
                    [self.samples[n, :, 0] >= self.xL -
                     self.xi[n, :] - cp.multiply(self.param[n], self.slack[n])]
                self.constraints_upp += \
                    [self.samples[n, :, 1] <= self.xU +
                     self.xi[n, :] + cp.multiply(self.param[n], self.slack[n])]
                
                if self.pareto:
                    self.constraints_upp += [self.samples[n,1,1] <= 
                     self.paretoBase - self.paretoSlope * self.samples[n,0,1] + 
                     cp.multiply(self.param[n], self.slack[n])]
                
            else:

                self.constraints_low += \
                    [self.samples[n, :] >= self.xL -
                     self.xi[n, :] - cp.multiply(self.param[n], self.slack[n])]
                self.constraints_upp += \
                    [self.samples[n, :] <= self.xU +
                     self.xi[n, :] + cp.multiply(self.param[n], self.slack[n])]

                if self.pareto:
                    self.constraints_upp += [self.samples[n,1] <= 
                     self.paretoBase - self.paretoSlope * self.samples[n,0] + 
                     cp.multiply(self.param[n], self.slack[n])]

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

        if self.pareto:
            paretoBase = self.paretoBase.value
            paretoSlope = self.paretoSlope.value
        else:
            paretoBase = None
            paretoSlope = None

        # Return solution
        sol = {
            'xL': self.xL.value,
            'xU': self.xU.value,
            'xi': self.xi.value,
            'constraints_low': self.constraints_low,
            'constraints_upp': self.constraints_upp,
            'paretoBase': paretoBase,
            'paretoSlope': paretoSlope
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

        num_interior = sum(interior_mask)

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

        # Compute IDs of the samples which are in the interior
        exterior_ids = self.sample_ids[~interior_mask]

        return sol, complexity, value, exterior_ids, num_interior


def compute_confidence_region(samples, args):
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
    
    beta = args.beta
    rho = args.rho_min
    increment_factor = args.rho_incr
    itermax = args.rho_max_iter

    Nsamples = len(samples)

    regions = {}

    # Initialize scenario optimization problem
    problem = scenarioProblem()
    
    problem.init_problem(samples, np.arange(Nsamples), args.pareto_pieces)

    # Initialize value of rho (cost of violation)
    i = 0
    exterior_ids = [None]
    
    # Do not solve problem if difference between max/min is very small (to
    # avoid solver issues)
    tres = 1e-3
    if samples.ndim == 3:
        samples_max = np.max(samples, axis=2)
        samples_min = np.min(samples, axis=2)
    else:
        samples_max = samples
        samples_min = samples
    
    compareAt = np.abs(np.max(samples_max, axis=0) - 
                       np.min(samples_min, axis=0)) > tres

    df_regions = pd.DataFrame()
    df_regions.index.names = ['Sample']
    df_regions_stats = pd.DataFrame(columns = ['rho','complexity','beta',
                                               'Pviolation','Psatisfaction'])

    while len(exterior_ids) > 0 and i < itermax:     

        sol, complexity, x_star, exterior_ids, num_interior = \
            problem.solve(compareAt, rho)

        # If complexity is the same (or even higher) as in previous iteration, 
        # skip or break
        if i > 0 and complexity >= regions[i-1]['complexity']:
            if complexity < 0.5*Nsamples:
                # If we are already at the outside of the slurf, break overall
                break
            else:
                # If we are still at the inside, increase rho and continue
                rho *= increment_factor
                continue
        
        # If complexity is equal to number of samples, the bound is not
        # informative (violation probability is one), so skip to next rho
        if complexity == Nsamples:
            rho *= increment_factor
            continue
        
        print('\nScenario problem solved of size {}; rho = {:0.4f}'.\
              format(problem.Nsamples, rho))
        print(' - Samples in interior of region:', num_interior)
        print(' - Complexity:', complexity)
        
        # Reinitialize problem only if we can reduce its size (i.e., if there
        # are samples fully in the interior of the current solution set)
        if num_interior > 0:
            problem.init_problem(samples[exterior_ids], exterior_ids,
                                 args.pareto_pieces)

        Pviolation = np.round(1 - etaLow(Nsamples, complexity, beta), 4)
        print(' - Upper bound on violation probability: {:0.6f}'.format(Pviolation))

        regions[i] = {
            'x_low': sol['xL'],
            'x_upp': sol['xU'],
            'rho': rho,
            'complexity': complexity,
            'Pviolation': Pviolation,
            'xi': sol['xi']
            }
        
        # Append results to dataframe
        df_regions['x_low'+str(i)] = sol['xL']
        df_regions['x_upp'+str(i)] = sol['xU']
        df_regions_stats.loc[i] = [rho, complexity, beta, Pviolation, 1-Pviolation]

        # Increment
        rho *= increment_factor
        i += 1

    return regions, df_regions, df_regions_stats