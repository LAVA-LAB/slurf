# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:49:38 2021

@author: Thom Badings
"""

import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def timePoints(T_end, N):
    '''
    Returns linspace of N steps until T_end
    '''
    
    return np.linspace(0, T_end, N)

def sampleRate(low, upp, size = 1):
    '''
    Get 'size' samples from the uniform probability distribution over failure 
    rates, over the interval [low,upp]
    '''
    
    r = np.random.uniform(low, upp, size)
    
    return np.maximum(0, r)

def rel(t, rate):
    '''
    Compute the reliability at time 't', under failure rate 'rate'
    '''
    
    return np.exp(-rate * t)

def penaltyBasedProgram(costOfRegret, R):
    '''
    Solve the penalty-based convex program, for a given cost of regret, and
    a given sample set R (of reliability curves)
    '''

    # Define convex optimization program
    x_mean  = cp.Variable(T_points-1, nonneg=True)
    x_width = cp.Variable(T_points-1, nonneg=True)
    
    # Define regret/slack variables
    xi      = cp.Variable(Nsamples, nonneg=True)
    
    # Cost of violation
    rho = cp.Parameter()
    rho.value = costOfRegret
    
    constraints_low = []
    constraints_upp = []
    
    # Add constraints for each samples
    for n in range(Nsamples):
        
        constraints_low += [R[n, 1:] >= x_mean - x_width - xi[n]]
        constraints_upp += [R[n, 1:] <= x_mean + x_width + xi[n]]
        
    obj = cp.Minimize( sum(x_width) + rho * sum(xi) )
        
    prob = cp.Problem(obj, constraints_low + constraints_upp)
    prob.solve( solver='GUROBI' )
    
    x_star = prob.value
    
    # Determine complexity of the solution
    complexity = 0
    for n in range(Nsamples):
        
        # If duel value of a constraint is nonzero, that constraint is active
        if any(np.abs(constraints_low[n].dual_value) > 1e-3):
            complexity += 1
            
        if any(np.abs(constraints_upp[n].dual_value) > 1e-3):
            complexity += 1
    
    return x_mean.value, x_width.value, complexity, x_star

#### Generate reliability samples

T_end = 10
T_points = 11

low, upp = 0.2, 2
T = timePoints(T_end, N = T_points)

Nsamples = 200

rates = sampleRate(low, upp, size = Nsamples)

R = np.zeros((Nsamples, len(T)))

fig, ax = plt.subplots()

for n in range(Nsamples):
    
    for t,time in enumerate(T):
        
        R[n, t] = rel(time, rates[n])
        
    # if Nsamples < 1000 or n % int(Nsamples/100) == 0:
    plt.plot(T, R[n, :], color='k', lw=0.3, ls='dotted', alpha=0.3)
    
plt.xlabel('Time')
plt.ylabel('Reliability')

ax.set_title('Reliability confidence region (N='+str(Nsamples)+' samples)')

#### Compute bounds on violation probability with standard scenario approach

d = (T_points-1) * 2        # Decision variables
eps = 0.1                  # Violation probability

# Compute confidence probability beta
beta_conv = betaF.cdf(eps, d, Nsamples-d+1)

print('Confidence level with traditional scenario approach:', beta_conv)

#### Compute bounds on violation probability with risk and complexity theory
# %%

# Set colors and markers
colors = sns.color_palette()
markers = ['o', '*', 'x', '.', '+']

# Values of rho (cost of regret) at which to solve the scenario program
rho_list = [2, 1, 0.5, 0.3, 0.1]

x_low = {}
x_upp = {}
violation_prob = np.zeros(len(rho_list))

for i,rho in enumerate(rho_list):

    x_mean, x_width, c_star, x_star = penaltyBasedProgram(rho, R)
    
    print("\nThe optimal value is", x_star)
    
    print('Complexity of penalty-based program:',c_star)
        
    from compute_eta import etaLow, betaLow
    
    beta_desired = 0.99
    
    violation_prob[i] = 1 - etaLow(Nsamples, c_star, beta_desired)
    
    print('Upper bound on violation probability:',violation_prob[i])
    
    confidence_prob = betaLow(Nsamples, c_star, 1-violation_prob[i])
    
    print('Error with desired beta:',np.abs(beta_desired - confidence_prob))
    
    # Plot confidence regions
    labelStr = r'$\rho$: '+str(rho)+ \
                '; $s^*$: '+str(c_star)+ \
                '; $\epsilon$: '+str(np.round(violation_prob[i], 2))
    
    x_low[i] = np.hstack((1, x_mean - x_width))
    x_upp[i] = np.hstack((1, x_mean + x_width))
    
    plt.plot(T, x_low[i], lw=2, marker=markers[i], 
             color=colors[i], label=labelStr)
    plt.plot(T, x_upp[i], lw=2, marker=markers[i], 
             color=colors[i])
    
plt.legend()
plt.show()

#### Validation of obtained bounds on the violation probability
# %%

def validateBounds(trials, x_low, x_upp):

    count = np.zeros(len(rho_list))
    
    rates_validate = sampleRate(low, upp, size = trials)
    
    for n in range(trials):
        
        R1 = np.zeros(len(T))
        
        # Generate another curve
        for t,time in enumerate(T):
            
            R1[t] = rel(time, rates_validate[n])
        
        for i,rho in enumerate(rho_list):
            
            # Check if sampled curve is within the interval for this value of rho
            if all( R1 >= x_low[i] ) and \
               all( R1 <= x_upp[i] ):
                   
               # Increment count
               count[i] += 1
               
    return count

trials  = 10000
count = validateBounds(trials, x_low, x_upp)
               
empirical_violation = 1 - count/trials
           
print('\n\nEmpirical violation probabilities:', np.round(empirical_violation, 3))

if all(empirical_violation < violation_prob):
    
    print('\n > GOOD: All bounds in the violation probability were correct!')
