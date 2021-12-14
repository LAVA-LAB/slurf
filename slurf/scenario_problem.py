import numpy as np
import cvxpy as cp

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
    	
        # Define convex optimization program
        x_mean  = cp.Variable(self.dim-1, nonneg=True)
        x_width = cp.Variable(self.dim-1, nonneg=True)
        
        # Define regret/slack variables
        xi      = cp.Variable(self.Nsamples, nonneg=True)
        
        # Cost of violation
        rho = cp.Parameter()
        rho.value = costOfRegret
        
        constraints_low = []
        constraints_upp = []
        
        # Add constraints for each samples
        for n in range(self.Nsamples):
            
            constraints_low += [self.samples[n, 1:] >= x_mean - x_width - xi[n]]
            constraints_upp += [self.samples[n, 1:] <= x_mean + x_width + xi[n]]
            
        obj = cp.Minimize( sum(x_width) + rho * sum(xi) )
            
        prob = cp.Problem(obj, constraints_low + constraints_upp)
        prob.solve( solver='GUROBI' )
        
        x_star = prob.value
        
        # Determine complexity of the solution
        complexity = 0
        for n in range(self.Nsamples):
            
            # If duel value of a constraint is nonzero, that constraint is active
            if any(np.abs(constraints_low[n].dual_value) > 1e-3):
                complexity += 1
                
            if any(np.abs(constraints_upp[n].dual_value) > 1e-3):
                complexity += 1
        
        return x_mean.value, x_width.value, complexity, x_star
        
        
        
       
