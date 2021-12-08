# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:49:38 2021

@author: Thom Badings
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def timePoints(T_end, N):
    
    return np.linspace(0, T_end, N)

def sampleRate(low, upp, size = 1):
    
    r = np.random.uniform(low, upp, size)
    
    return np.maximum(0, r)

def rel(t, rate):
    
    return np.exp(-rate * t)

from scipy.stats import beta as betaF

T_end = 10
T_points = 11

low, upp = 0.3, 1
T = timePoints(T_end, N = T_points)

Nsamples = 200

rates = sampleRate(low, upp, size = Nsamples)

R = np.zeros((Nsamples, len(T)))

fig, ax = plt.subplots()

for n in range(Nsamples):
    
    for t,time in enumerate(T):
        
        R[n, t] = rel(time, rates[n])
        
    # if Nsamples < 1000 or n % int(Nsamples/100) == 0:
    plt.plot(T, R[n, :], color='k', lw=0.3)
    
plt.xlabel('Time')
plt.ylabel('Reliability')
    
# %%

# Compute guarantee on SLURF predictor

## With standard scenario approach theory

d = (T_points-1) * 2        # Decision variables
eps = 0.05                  # Violation probability

# Compute confidence probability beta
beta_conv = betaF.cdf(eps, d, Nsamples-d+1)

print('Confidence level with traditional scenario approach:', beta_conv)

# %%
## With risk and complexity scenario approach theory

# Define convex optimization program
x_mean  = cp.Variable(T_points-1, nonneg=True)
x_width = cp.Variable(T_points-1, nonneg=True)

# Define regret/slack variables
xi      = cp.Variable(Nsamples, nonneg=True)

# Cost of violation
rho = cp.Parameter()
rho.value = 0.05

constraints_low = []
constraints_upp = []

# Add constraints for each samples
for n in range(Nsamples):
    
    constraints_low += [R[n, 1:] >= x_mean - x_width - xi[n]]
    constraints_upp += [R[n, 1:] <= x_mean + x_width + xi[n]]
    
obj = cp.Minimize( sum(x_width) + rho * sum(xi) )
    
prob = cp.Problem(obj, constraints_low + constraints_upp)
prob.solve( solver='GUROBI' )

print("\nThe optimal value is", prob.value)

complexity = 0
for n in range(Nsamples):
    
    if any(np.abs(constraints_low[n].dual_value) > 1e-3):
        complexity += 1
        print('Sample',n,'active for lower bound')
        
    if any(np.abs(constraints_upp[n].dual_value) > 1e-3):
        complexity += 1
        print('Sample',n,'active for upper bound')

from compute_eta import betaLow

beta_RC = betaLow(Nsamples, complexity, 1-eps)

print('Confidence level with Risk & Complexity theory:', beta_RC)

# Plot confidence regions
plt.plot(T, np.hstack((1, x_mean.value - x_width.value)), color='red')
plt.plot(T, np.hstack((1, x_mean.value + x_width.value)), color='red')

plt.show()