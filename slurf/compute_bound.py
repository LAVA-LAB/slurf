#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:35:21 2021

@author: thom
"""

import numpy as np
import math

def etaLow(N, k, beta, verbose=False):
    '''
    Rootfinding problem to compute the lower bound on the satprob (eta)

    Parameters
    ----------
    N : int
        Number of samples.
    k : int
        Complexity of the problem (is 1 + number of discarded samples).
    beta : float
        Confidence probability.

    Returns
    -------
    eta_star : float
        Lower bound on the satisfaction probability.

    '''
    
    # Compute combination (i, k) in logarithmic base (term 1)
    m1 = np.array([np.arange(k,N+1)])

    rep1 = np.repeat(m1, N-k+1, axis=0)
    log1 = np.array([[np.log(val) if val != 0 else 0 for val in row]
                     for row in rep1 ])
    aux1 = np.sum( np.triu(log1, 1), axis=1 )
    
    rep2 = np.repeat(m1-k, N-k+1, axis=0)
    log2 = np.array([[np.log(val) if val != 0 else 0 for val in row]
                     for row in rep2 ])
    aux2 = np.sum( np.triu(log2, 1), axis=1)
    
    coeffs1 = aux2 - aux1
    
    # Initial guess for value of t (for lower bound epsilon)
    t1 = 1E-9
    t2 = 1-1E-9
    
    while t2-t1 > 1E-10:
        t = (t1+t2)/2
        polyt = 1+(1-beta)/(N) - (1-beta)/(N)*np.sum( np.exp( coeffs1 - (N-m1)*np.log(t) ), axis=1 )
        
        if polyt > 0:
            t2 = t
        else:
            t1 = t
    
    eta_star = t1
    
    if verbose:
        print('Compute remainder..')
        print('Remainder is:',remainder(k,N,eta_star,beta))
    
    return eta_star

def remainder(k,N,eta,beta):
    
    return math.comb(N,k) * eta**(N-k) - (1-beta)/N * sum([math.comb(i,k)*eta**(i-k) for i in range(k,N)])

def compute_beta(k,N,eta):
    
    invbeta = 0
    
    for i in range(k, N):
        
        log0 = math.log(eta**(i-N))
        log1 = math.log(math.factorial(i)) + math.log(math.factorial(N-k))
        log2 = math.log(math.factorial(N)) + math.log(math.factorial(i-k))
        
        invbeta += np.exp( log0 + log1 - log2 )
    
    beta = 1 - N/invbeta
    
    return beta

# Approximation of binomial cdf with continuity correction for large n
# n: trials, p: success prob, m: starting successes
def BCDF(p, n, m):
    return 1-CDF((m-0.5-(n*p))/math.sqrt(n*p*(1-p)))
def CDF(x):
    return (1.0 + math.erf(x/math.sqrt(2.0)))/2.0

def determine_discarded(N=1000, beta=1e-6, eta=0.86):
    
    k = 0
    res = -1
    
    while res < beta and k < N:
        #res = math.comb(N+1, N) * sum([math.comb(N, i)*(eta)**(N-i)*(1-eta)**(i) for i in range(k+1)])
        res = sum([math.comb(N, i)*(eta)**(N-i)*(1-eta)**(i) for i in range(k+1)])
    
        k+=1
    
        # print(k,':',res)
        
    print('Number of discarded constraints:',k)