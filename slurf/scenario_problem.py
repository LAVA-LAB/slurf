import numpy as np
import cvxpy as cp
import pandas as pd

from slurf.compute_bound import etaLow


class scenarioProblem:
    """
    Functions related to the scenario optimization part.
    """

    def init_problem(self, samples, sample_ids, paretoP=0, paretoCost=1):
        '''
        
        Parameters
        ----------
        samples Current subset of samples active in optimization problem
        sample_ids Indices of sample subset (compared to complete sample set)
        paretoP Number of linear pieces in the Pareto-front
        paretoCost Coefficient in the Pareto-front portion of the objective
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

        # Check if number of Pareto-front linear pieces is nonzero
        if paretoP > 0: 
            if self.dim != 2:
                print('ERROR: Can currently only solve for Pareto-front in 2D')
                assert False
            
            self.paretoBase = cp.Variable(paretoP, nonneg=True)
            
            # Set the different possible sloped of the Pareto-front
            diff = np.max(samples, axis=0) - np.min(samples, axis=0)
            coeff = diff[1] / diff[0]
            self.paretoSlope = coeff * np.array([2**(i-(paretoP-1)/2) 
                                                 for i in range(paretoP)])
            
            self.pareto = True
        else:
            self.pareto = False

        # Define convex optimization program
        self.xU = cp.Variable(self.dim, nonneg=True)
        self.xL = cp.Variable(self.dim, nonneg=False)

        # Define regret/slack variables
        self.xi = cp.Variable((self.Nsamples, self.dim), nonneg=True)

        # Cost of violation
        self.rho = cp.Parameter()

        # Define slack variables (to enable/disable individual constraints)
        self.param = cp.Parameter(self.Nsamples)
        self.slack = cp.Variable(self.Nsamples)

        self.constraints_base = [self.xU >= self.xL]
        self.constraints_low  = [None for n in range(self.Nsamples)]
        self.constraints_upp  = [None for n in range(self.Nsamples)]
        if self.pareto:
            self.constraints_par = [None for n in range(self.Nsamples)]
        else:
            self.constraints_par = []

        # Add constraints
        for n in range(self.Nsamples):

            # Switch between precise vs. imprecise samples
            if imprecise:
                
                # For imprecise samples, formulate constraint such that the
                # whole box sample is contained in the confidence region
                self.constraints_low[n] = self.samples[n, :, 0] >= self.xL - \
                     self.xi[n, :] - cp.multiply(self.param[n], self.slack[n])
                self.constraints_upp[n] = self.samples[n, :, 1] <= self.xU + \
                     self.xi[n, :] + cp.multiply(self.param[n], self.slack[n])
                
                if self.pareto:
                    self.constraints_par[n] = (self.paretoSlope * 
                         (self.samples[n,0,1] - self.xi[n,0]) + 
                         (self.samples[n,1,1] - self.xi[n,1]) <=
                            self.paretoBase + cp.multiply(self.param[n], 
                                                          self.slack[n]))
                
            else:

                self.constraints_low[n] = self.samples[n, :] >= self.xL - \
                     self.xi[n, :] - cp.multiply(self.param[n], self.slack[n])
                self.constraints_upp[n] = self.samples[n, :] <= self.xU + \
                     self.xi[n, :] + cp.multiply(self.param[n], self.slack[n])

                if self.pareto:
                    self.constraints_par[n] = (self.paretoSlope * 
                         (self.samples[n,0] - self.xi[n,0]) + 
                         (self.samples[n,1] - self.xi[n,1]) <=
                            self.paretoBase + cp.multiply(self.param[n], 
                                                          self.slack[n]))

        # Objective function
        if self.pareto:
            self.constraints_base += [self.xL == np.min(samples, axis=0) - 0.1]
            
            self.obj = cp.Minimize(sum(self.xU - self.xL) +
                               self.rho * cp.sum(self.xi) +
                               paretoCost * sum(self.paretoBase))
            
        else:             
            self.obj = cp.Minimize(sum(self.xU - self.xL) +
                               self.rho * cp.sum(self.xi))

        constraints = self.constraints_low + self.constraints_base + \
                        self.constraints_upp + self.constraints_par

        self.prob = cp.Problem(self.obj, constraints)

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
            pareto = {
                'base': self.paretoBase.value,
                'slope': self.paretoSlope,
                }
        else:
            pareto = {}

        # Return solution
        sol = {
            'xL': self.xL.value,
            'xU': self.xU.value,
            'xi': self.xi.value,
            'constraints_low': self.constraints_low,
            'constraints_upp': self.constraints_upp,
            'constraints_par': self.constraints_par,
            'pareto': pareto,
            'halfspaces': self.get_halfspaces()
            }

        return self.prob.value, sol
    
    def get_halfspaces(self):
        '''
        Set halfspace matrix based on the given `base` and `slope` (both 1D arrays)
        '''
        
        if self.dim != 2:
            return None
        
        G = np.array([[-1, 0, self.xL.value[0]],
                      [0, -1, self.xL.value[1]],
                      [1, 0, -self.xU.value[0]],
                      [0, 1, -self.xU.value[1]]])
        
        if self.pareto:
            
            H = np.vstack((self.paretoSlope, np.ones(len(self.paretoSlope)), 
                           -self.paretoBase.value)).T
            G = np.vstack((G, H))
        
        return G
        

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

            if self.pareto:
                pareto_dual = sol['constraints_par'][i].dual_value
                pareto_dual_check = all(np.isclose(pareto_dual, 0))
            else:
                pareto_dual_check = True

            # If all dual values are zero (and xi=0), sample is in interior
            if all(np.isclose(LB_dual, 0)) and all(np.isclose(UB_dual, 0)) and \
                pareto_dual_check and all(np.isclose(sol['xi'][i], 0)):

                interior_mask[i] = True
                support_mask[i] = False

        num_interior = sum(interior_mask)

        disable_mask = np.array(~support_mask, dtype=int)
        _, solC = self.solve_instance(disable_mask, costOfRegret)

        for i in range(self.Nsamples):

            # If current sample is in interior, or is relaxed, we can skip it
            if not support_mask[i]:
                continue
            
            # If any xi_i is nonzero, sample is of support by construction
            if any(sol['xi'][i] > 0.01):                
                continue

            # Disable the samples that are: not of support and are not relaxed
            disable_mask = np.array(~support_mask, dtype=int)
            disable_mask[i] = 1

            # Solve the optimization problem
            _, solB = self.solve_instance(disable_mask, costOfRegret)

            if self.pareto:
                pareto_check = all( np.isclose(sol['pareto']['base'], 
                                               solB['pareto']['base']) )
            else:
                pareto_check = True

            # If the objective has not changed, this sample is not of support
            if all( (np.isclose(sol['xL'], solB['xL']))[compareAt] ) and \
               all( (np.isclose(sol['xU'], solB['xU']))[compareAt] ) and \
               pareto_check:

                support_mask[i] = 0

            del disable_mask

        # Compute complexity and store critical sample set
        complexity = sum(support_mask)
        critical_set = self.sample_ids[support_mask]

        # Compute IDs of the samples which are in the interior
        exterior_ids = self.sample_ids[~interior_mask]

        return sol, complexity, critical_set, exterior_ids, num_interior



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

    print('- Number of dimensions below threshold:',sum(~compareAt))

    df_regions = pd.DataFrame()
    df_regions.index.names = ['Sample']
    
    cols = ['rho','complexity'] + ['satprob_beta='+str(b) for b in beta]
    df_regions_stats = pd.DataFrame(columns = cols)

    while len(exterior_ids) > 0 and i < itermax:     

        sol, complexity, critical_set, exterior_ids, num_interior = \
            problem.solve(compareAt, rho)

        # If complexity is the same (or even higher) as in previous iteration, 
        # skip or break
        if i > 0 and complexity >= regions[i-1]['complexity']:
            if rho > 2: #complexity < 0.5*Nsamples:
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
        # BUT: do not do this step in case of pareto plot
        if args.pareto_pieces == 0 and num_interior > 0:
            problem.init_problem(samples[exterior_ids], exterior_ids,
                                  args.pareto_pieces)

        regions[i] = {
            'x_low': sol['xL'],
            'x_upp': sol['xU'],
            'rho': rho,
            'complexity': complexity,
            'critical_set': critical_set,
            'xi': sol['xi'],
            'pareto': sol['pareto'],
            'halfspaces': sol['halfspaces']
            }
        
        Psat = []
        for b in beta:

            Pviolation = np.round(1 - etaLow(Nsamples, complexity, b), 4)
            Psat += [1 - Pviolation]
            regions[i]['satprob_beta='+str(b)] = 1 - Pviolation
            print(' - Lower bound on sat.prob for beta={}: {:0.6f}'.\
                  format(b, 1-Pviolation))
                
        regions[i]['eta_series'] = pd.Series(Psat, index=beta)
        
        # Append results to dataframe
        df_regions['x_low'+str(i)] = sol['xL']
        df_regions['x_upp'+str(i)] = sol['xU']
        df_regions_stats.loc[i] = [rho, complexity] + Psat

        # Increment
        rho *= increment_factor
        i += 1

    return regions, df_regions, df_regions_stats