# -*- coding: utf-8 -*-
"""
Read a CSV file for the average lower bounds on the containment probability
and convert this table to the corresponding table with only the average values
and their standard deviation. This results in tables such as Table 2 in
Badings et al. (2022), CAV 2022.
"""

import numpy as np
import pandas as pd
import argparse
from ast import literal_eval

# Parse argument that gives the file to read from
parser = argparse.ArgumentParser(description="Generate average lower bounds table")
parser.add_argument('--file', type=str, action="store", dest='file', 
                    default=None, help="File to load and convert")
parser.add_argument('--outfile', type=str, action="store", dest='outfile', 
                    default=None, help="Name of file to export table to")


args = parser.parse_args()    

# Read CSV file
df = pd.read_csv(args.file, sep=';')

# Retrieve values of the confidence level from the table
betas = list(df.drop(columns=['#', 'N', 'rho', 'seed', 'Frequentist']).columns)

# Iterate over all values of N (sample size) and rho (cost of violation)
Ns   = np.unique( df['N'] )
rhos = np.unique( df['rho'] )

betas_mean = [str(b)+'_mean' for b in betas]
betas_std  = [str(b)+'_std' for b in betas]
df_res = pd.DataFrame()
df_res.index.name = 'N'

for N in Ns:
    for rho in rhos:
        print(' - Convert for N =',N,'and rho =',rho)
        
        # Filter for current N and rho
        current = df.loc[(df['N']==N) & (df['rho']==rho)]
        
        # Get mean and standard deviation for every confidence level
        for beta in betas:
        
            df_res.loc[N, str(beta)+'_mean'] = np.round(current[str(beta)].mean(), 3)
            df_res.loc[N, str(beta)+'_std'] = np.round(current[str(beta)].std(), 3)
            
        # Get frequentist mean and standard deviation
        df_res.loc[N, 'Frequentist_mean'] = np.round(current['Frequentist'].mean(), 3)
        df_res.loc[N, 'Frequentist_std'] = np.round(current['Frequentist'].std(), 3)
        
print('\nTable of average (+ std.dev.) lower bounds per confidence level:')
print(df_res)
df_res.to_csv(args.outfile, sep=';')