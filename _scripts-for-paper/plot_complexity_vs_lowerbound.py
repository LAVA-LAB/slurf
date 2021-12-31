# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:27:14 2021

@author: Thom Badings
"""

import numpy as np
import pandas as pd
from slurf.compute_bound import etaLow

def create_plot_csv(Nlist, betaList, file):
    
    for N in Nlist:
        
        betaStrList = ['b'+str(beta) for beta in betaList]
        cList = np.arange(0, N)
        df = pd.DataFrame(index=cList, columns=betaStrList)
        
        for c in cList:
            print('- Compute for N='+str(N)+' and complexity c='+str(c))
            
            for beta,betaStr in zip(betaList,betaStrList):
                df[betaStr][c] = etaLow(N, c, beta)
                
        df.to_csv(file+'_N='+str(N)+'.csv', sep=';')
    
create_plot_csv([10,25,50,100,250], [0.9, 0.99, 0.999], 'satprob-bounds')