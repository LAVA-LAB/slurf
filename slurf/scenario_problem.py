import numpy as np
import cvxpy as cp
import pandas as pd
import copy
import time

from slurf.compute_bound import etaLow
from slurf.commons import intersect
from slurf.solution_sampler import refine_solutions
from slurf.export import plot_results

class scenarioProblem:
    """
    Functions related to the scenario optimization part.
    """

    def init_problem(self, refine, samples, sample_ids, paretoP=0, 
                     paretoCost=1):
        """
        Initialize scenario optimization problem
        
        Parameters
        ----------
        :refine: Boolean whether solution vectors should be refined or not
        :samples: Current subset of samples active in optimization problem
        :sample_ids: Indices of sample subset (compared to complete sample set)
        :paretoP: Number of linear pieces in the Pareto-front
        :paretoCost: Coefficient in the Pareto-front portion of the objective
        ----------

        """
        
        self.refine = refine
        self.samples = samples
        self.sample_ids = sample_ids
        
        # Check if imprecise samples are used (means that dimension = 3)
        if self.samples.ndim == 3:
            self.exact = False
            self.Nsamples, self.dim, _ = np.shape(self.samples)
        else:
            self.exact = True
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

        # Cost of relaxation
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
            if not self.exact:
                
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

    def solve_instance(self, disable_mask, costOfRelaxation, solver='ECOS'):
        """
        Solve a single instance of the scenario optimization problem
        
        Parameters
        ----------
        :disable_mask: Mask to disable solutions that can be left out of the 
        optimization problem
        :costOfRelaxation: Current value of the cost of relaxation (rho)
        :solver: Solver to use (optional; ECOS by default, which is provided
        which CVXPY)
        ----------
        
        Returns the optimal value of the solution, and the solution itself.

        """
        
        # Set current parameters
        self.param.value = disable_mask
        self.rho.value = costOfRelaxation

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
        """
        Set halfspace matrix based on the given `base` and `slope` 
        (both 1D arrays)
        """
        
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
      
    def solve(self, sampleObj, compareAt, costOfRelaxation=1, 
              all_solutions=None):
        """
        Solve the scenario optimization for a specific value of the cost of
        relaxation, and determine the resulting complexity.
        
        Parameters
        ----------
        :sampleObj: Object containing the solution vectors
        :compareAt: (Time) values to solve the problem for
        :costOfRelaxation: Current value of the cost of relaxation (rho)
        :solver: Object of all solutions (used for approximate model checking 
                                          only)
        ----------
        
        Returns
        ----------
        :sol: Solution to the optimization problem
        :complexity: Complexity of the scenario optimization problem
        :critical_set: Set of solutions that are of complexity
        :exterior_ids: Solution IDs that are being relaxed (out of the obtained
                                                            confidence region)
        :num_interior: Number of solutions in the interior of the conf. region
        :refine_set: Set of solutions to refine (in case of approximate mode)
        ----------

        """
        
        # Solve initial problem with all constraints
        mask = np.zeros(self.Nsamples)
        value, sol = self.solve_instance(mask, costOfRelaxation)

        # If upper bound equals lower bound, we can already conclude that
        # the complexity equals the number of samples
        if self.exact and not self.pareto and \
          all(np.isclose(sol['xL'], sol['xU'])):
            print(' >> Solution is a point, so skip to next problem iteration')
            
            return sol, len(self.samples), self.sample_ids, \
                self.sample_ids, 0, None

        # Initialize masks
        interior_mask = np.zeros(self.Nsamples, dtype=bool)
        support_mask = np.ones(self.Nsamples, dtype=bool)

        if self.exact:
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
                if all(np.isclose(LB_dual, 0)) and all(np.isclose(UB_dual, 0))\
                    and pareto_dual_check and all(np.isclose(sol['xi'][i], 0)):
    
                    interior_mask[i] = True
                    support_mask[i] = False

        num_interior = sum(interior_mask)

        disable_mask = np.array(~support_mask, dtype=int)
        _, solC = self.solve_instance(disable_mask, costOfRelaxation)

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
            _, solB = self.solve_instance(disable_mask, costOfRelaxation)

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
        
        critical_set = self.sample_ids[support_mask]
        
        if self.exact:
            complexity = sum(support_mask)
            boundary_set = []
            refine_set = None
        
        else:
            boundary_mask = np.zeros(self.Nsamples, dtype=bool)
            
            for i in range(self.Nsamples):
                    
                # If any xi_i is zero but the solution is in the support mask,
                # it must be on the boundary of the confidence region
                if any(np.isclose(sol['xi'][i], 0)) and support_mask[i]:
                    boundary_mask[i] = True
    
            refine_mask = copy.copy(boundary_mask)
            boundary_set = self.sample_ids[boundary_mask]
            print( 'Samples on boundary:', boundary_set )
            intersect_mask = np.zeros(len(all_solutions), dtype=bool)
            
            for j in np.where(boundary_mask)[0]:
                
                for i,I in enumerate(intersect_mask):
                    
                    # If already on the boundary, set to True and skip
                    if i in boundary_set:
                        #intersect_mask[i] = True
                        continue
                    
                    # Otherwise, check if sample intersects with (affine 
                    # extensions of) the boundary
                    if any([intersect(all_solutions[i,d,:],self.samples[j,d,:]) 
                            for d in range(self.dim)]):
                        intersect_mask[i] = True
                        
                        if self.refine:
                            # Select this solution on the boundary for refinement
                            refine_mask[j] = True
                            
                            # If this solution was already refined, than also
                            # refine the one it intersects with
                            if sampleObj[self.sample_ids[j]]._refined:
                                print(' - Sample',j,'already refined, so refine sample',i)
                                
                                refine_mask[i] = True
                        
            
            refine_set = np.union1d(self.sample_ids[refine_mask], boundary_set)
                    
            intersect_set = np.where(intersect_mask)[0]
            # print('Samples intersecting with the boundary:', intersect_set)
            
            complexity = len(np.union1d(critical_set, intersect_set))

        # Compute IDs of the samples which are in the interior
        exterior_ids = self.sample_ids[~interior_mask]

        return sol, complexity, critical_set, exterior_ids, num_interior, \
                refine_set


def init_rho_list(args):
    """
    Define the list of the costs of relaxation to use by SLURF

    Parameters
    ----------
    :args: Argument given by parser
    ----------

    Returns
    ----------
    :Sorted: list of the values for the cost of relaxation to use
    ----------

    """
    
    if args.rho_list:
        rho_list = args.rho_list
    else:
        # If no list of values
        ls = np.minimum(int(args.Nsamples/2), 
                        [0, 1, 2, 4, 6, 8, 10, 15, 20, 50, 100, 200, 400])
        
        rho_list = np.round([1/(int(n)+0.5) for n in ls], 3)
        
    return np.unique(np.sort(rho_list))


def compute_confidence_region(samples, beta, args, rho_list, sampleObj=None):
    """
    Compute the confidence region for a given set of solution vectors, a given
    confidence probability (beta), and a list of cost of relaxations (rho_list)

    Parameters
    ----------
    :samples: 2D Numpy array of samples
    :beta: Confidence probability (close to one means a good conf. level)
    :args: Argument given by parser
    :rho_list: List of numeric values for the cost of relaxation
    :sampleObj: Object containing the solution vectors (used for refinement,
                                                        if enabled; optional)
    ----------

    Returns
    ----------
    :regions: Dictionary of results per rho
    :df_regions:, :df_regions_stats: Pandas DFs with stats and results
    :refineID: IDs of the solution vectors that are to be refined
    
    ----------

    """
    
    Nsamples = len(samples)
    regions = {}

    # Initialize scenario optimization problem
    problem = scenarioProblem()
    problem.init_problem(args.refine, samples, np.arange(Nsamples), 
                         args.pareto_pieces)

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

    refineID = np.array([], dtype=int)

    for i,rho in enumerate(rho_list):    

        sol, complexity, critical_set, exterior_ids, num_interior, \
            refine_set = problem.solve(sampleObj, compareAt, rho, 
                                       all_solutions=samples)
        
        print('\nScenario problem solved of size {}; rho = {:0.4f}'.\
              format(problem.Nsamples, rho))
        print(' - Samples in interior of region:', num_interior)
        print(' - Complexity:', complexity)
        
        # Reinitialize problem only if we can reduce its size (i.e., if there
        # are samples fully in the interior of the current solution set)
        # BUT: do not do this step in case of pareto plot
        if args.pareto_pieces == 0 and num_interior > 0:
            problem.init_problem(args.refine, samples[exterior_ids], 
                                 exterior_ids, args.pareto_pieces)

        # Store results about the confidence region for the current conf.prob.
        regions[i] = {
            'x_low': sol['xL'],
            'x_upp': sol['xU'],
            'rho': rho,
            'complexity': complexity,
            'critical_set': critical_set,
            'xi': sol['xi'],
            'pareto': sol['pareto'],
            'halfspaces': sol['halfspaces'],
            'refine_set': refine_set
            }
        
        # If not in exact mode, also determine solutions to be refined
        if not args.exact:
            refineID = np.union1d(refineID, refine_set)
        
        # Determine the satisfaction/containment probability for the current
        # confidence region
        Psat = []
        for b in beta:

            time_start = time.process_time()
            Pviolation = np.round(1 - etaLow(Nsamples, complexity, b), 6)
            time_taken = time.process_time() - time_start
            Psat += [1 - Pviolation]
            regions[i]['satprob_beta='+str(b)] = 1 - Pviolation
            print(' - Lower bound on sat.prob for beta={}: {:0.6f}'.\
                  format(b, 1-Pviolation))
            print(' - Computing this bound took: {:0.3f}'.format(time_taken))
                
        regions[i]['eta_series'] = pd.Series(Psat, index=beta)
        
        # Append results to dataframe
        df_regions['x_low'+str(i)] = sol['xL']
        df_regions['x_upp'+str(i)] = sol['xU']
        df_regions_stats.loc[i] = [rho, complexity] + Psat

    return regions, df_regions, df_regions_stats, refineID


def refinement_scheme(output_path, sampler, sampleObj, solutions, args, 
                      rho_list, plotEvery=1000, max_iter = 10):
    """
    Refinement scheme that refines imprecise solutions
    
    Parameters
    ----------
    :output_path: Path to folder for exporting results/figures
    :sampler: Sampler object
    :sampleObj: Object containing the solution vectors (used for refinement,
                                                        if enabled);
    :solutions: Solution vectors
    :args: Argument given by parser
    :rho_list: List of numeric values for the cost of relaxation
    :plotEvery: Determines how often we plot intermediate refinement results
    :maxIter: Maximum number of refinement iterations
    ----------

    Returns
    ----------
    :regions: Dictionary of results per rho
    :dfs_regions:, :dfs_regions_stats: Pandas DFs with stats and results
    :refined_df: Pandas DF with the refined solution vectors for each iteration
    
    """
    
    i = 0
    done = False
    
    args.no_refined = 0
    refined_df = pd.DataFrame()
    
    # Iterate while not finished with refining yet
    while not done:
        i += 1
        
        # Compute confidence region for current iteration
        regions, dfs_regions, dfs_regions_stats, refineID = \
            compute_confidence_region(solutions, args.beta, args, rho_list, 
                                      sampleObj)
        
        # If maximum number of iteration is exceeded, break
        if i > max_iter:
            break
        
        # Occasionally plot (intermediate) results
        if i % plotEvery == 0:
            file_suffix = 'refine'+str(i)
            
            plot_results(output_path, args, regions, solutions, file_suffix)
        
        toRefine = [r for r in refineID if not sampleObj[r].is_refined()]
        
        print(' - Refine samples:', toRefine)
        
        if len(toRefine) > 0:            
            solutions = refine_solutions(sampler, sampleObj, solutions, 
                                         toRefine, args.refine_precision, {})
        elif i > 1:
            done = True
            
        for n in toRefine:
            sampleObj[n]._refined = True
            args.no_refined += 1
        
        
        for z,region in regions.items():
            for beta,eta in region['eta_series'].items():
                refined_df.loc[i, str(z)+'_'+str(beta)] = eta
                
            refined_df.loc[i, str(z)+'_x_low'] = str(region['x_low'])
            refined_df.loc[i, str(z)+'_x_upp'] = str(region['x_upp'])
    
    return regions, dfs_regions, dfs_regions_stats, refined_df


def compute_confidence_per_dim(solutions, args, rho_list):
    """
    Perform scenario optimization for each measure/dimension individually.
    Used for performing the comparison with a 'naive baseline', as reported
    in the experiments of the main paper.
    
    Returns the satisfaction probability for a given set of solution vectors,
    a given confidence probability, and for every value in a list of cost of 
    relaxations.
    """
    
    dims = solutions.shape[1]
    beta = list(1-(1 - np.array(args.beta))/dims)
    
    satprobs = np.zeros((len(beta), dims))
    satprob  = np.ones(len(beta))
    
    for dim in range(solutions.shape[1]):
        reg, _, _, _, = compute_confidence_region(solutions[:,[dim]], beta, args, rho_list)
        
        satprobs[:,dim] = reg[0]['eta_series'].to_numpy()
        
        satprob *= satprobs[:,dim]
        
    return satprob