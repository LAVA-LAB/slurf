# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:27:14 2021

@author: Thom Badings
"""

import numpy as np
import pandas as pd
from slurf.compute_bound import etaLow

def create_plot_csv(Nlist, betaList, file):
    
    df = {}
    
    for N in Nlist:
        
        betaStrList = ['b'+str(beta) for beta in betaList]
        cList = np.arange(0, N)
        df[N] = pd.DataFrame(index=cList, columns=betaStrList)
        
        for c in cList:
            print('- Compute for N='+str(N)+' and complexity c='+str(c))
            
            for beta,betaStr in zip(betaList,betaStrList):
                df[N][betaStr][c] = etaLow(N, c, beta)
                
        df[N].to_csv(file+'_N='+str(N)+'.csv', sep=';')
        
    return df
    
df = create_plot_csv([200], [0.9, 0.99, 0.999], 'satprob-bounds')