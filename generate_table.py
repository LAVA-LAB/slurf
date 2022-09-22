# -*- coding: utf-8 -*-
"""
Grab all .json files in a certain folder and generate the benchmark statistics
table as presented in Badings et al. (2022), CAV 2022.
"""

import numpy as np
import pandas as pd
import argparse
import os
import glob, json
from slurf.util import path
from pathlib import Path

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

# Parse argument that gives the file to read from
parser = argparse.ArgumentParser(description="Generate benchmark statistics table")

parser.add_argument('--folder', type=str, action="store", dest='folder', 
                    default=None, help="Folder to read files from")
parser.add_argument('--outfile', type=str, action="store", dest='outfile', 
                    default=None, help="Name of file to export table to")
parser.add_argument('--mode', type=str, action="store", dest='mode', 
                    default=None, help="Is either 'statistics', 'lower_bounds', or 'scen_opt_time'")

args = parser.parse_args()    

# Set root directory
root_dir = os.path.dirname(os.path.abspath(__file__))

json_folder = path(root_dir, args.folder, '')

print('- Search for json files in:', json_folder)

json_pattern = os.path.join(json_folder, '*.json')
file_list = glob.glob(json_pattern)

print('- Number of json files found:', len(file_list))

# Get all JSON files in this folder
dfs = []
json_bounds = {}
json_montecarlo = {}

for i,file in enumerate(file_list):
    # Interpret as DataFrame
    with open(file) as f:
        json_data = pd.json_normalize(json.loads(f.read()))
        json_data['site'] = file.rsplit("/", 1)[-1]
        
        json_data['ID'] = i
    # Get nested dictionaries for lower bound values
    with open(file) as f:
        data = json.load(f)
        json_bounds[i] = data['lower_bound']
        json_montecarlo[i] = data['naive_bounds_max_rho']
        
    dfs.append(json_data)
df = pd.concat(dfs)

instances = df['name'].unique()
instantiators = df['instantiator'].unique()
checkers = df['model_checking_type'].unique()

TABLE = pd.DataFrame()

if args.mode == 'statistics':
    # Generate benchmark statistics table 
    
    TABLE.index.name = 'instance'
    
    for i in instances:
        for j in instantiators:
            for k in checkers:
                
                rows = df.loc[(df['name']==i) & (df['instantiator']==j) & (df['model_checking_type']==k)]
                
                if len(rows) > 0:
                    print('--- Collect all instances for:',i,j,k,'...')
                    
                    # Check if model size is consistent
                    if is_unique(rows['no_properties']):
                        TABLE.loc[i, 'no_properties'] = rows['no_properties'].to_numpy()[0]
                    if is_unique(rows['no_pars']):
                        TABLE.loc[i, 'no_pars'] = rows['no_pars'].to_numpy()[0]
                    if is_unique(rows['no_states']):
                        TABLE.loc[i, 'no_states'] = rows['no_states'].to_numpy()[0]
                    if is_unique(rows['no_trans']):
                        TABLE.loc[i, 'no_trans'] = rows['no_trans'].to_numpy()[0]
                       
                    # Grab more statistics
                    TABLE.loc[i, 'time_init'] = rows['time_init'].mean()
                    TABLE.loc[i, 'time_sample_x100'] = rows['time_sample_x100'].mean()
                    
                    Ns = np.sort(rows['N'])
                    scen_opt_times = rows['scen_opt_time'].to_numpy()
                    
                    # Add scenario optimization run times per value of N
                    for N,time in zip(Ns, scen_opt_times):
                        col = 'scen_opt_time_N='+str(N)
                        TABLE.loc[i, col] = time
                        
                        
elif args.mode == 'lower_bounds':
    # Table with lower bounds on the containment prob. (Table 2 in paper)
    
    TABLE.index.name = 'Nsamples'
    
    if not (is_unique(df['name']) and is_unique(df['instantiator'])
            and is_unique(df['model_checking_type'])):
        print('Error: Benchmark instance not consistent, so cannot create table of lower bounds')
        assert False
        
    rho = -1
    for b in json_bounds.values():
        if len(b) > 1:
            print('Error: Cannot create this table when multiple values of rho (cost of relaxation) are used')
            assert False
            
        if rho == -1:
            rho = list(b.keys())[0]
        else:
            if rho != list(b.keys())[0]:
                print('Error: Cannot create this table when the value of rho (costs of relaxation) differs between instances')
                assert False
        
    # Loop over numbers of samples
    Ns = np.sort(df['N'].unique())
    
    for N in Ns:
        
        sub_df = pd.DataFrame()
        
        rows = df.loc[df['N'] == N]
        # Loop over filtered set of rows
        for ID in rows['ID']:
            
            # Add bounds to the sub DataFrame
            sub_df = sub_df.append(json_bounds[ID][rho], ignore_index=True)
        
        # Compute average and add to table
        cols = sub_df.columns
        for col in cols:
            TABLE.loc[N, str(col)+'_mean'] = sub_df[col].mean()
            TABLE.loc[N, str(col)+'_stdev'] = sub_df[col].std()
        
    
elif args.mode == 'scen_opt_time':
    # Table with scenario optimization run times (Table 3 in paper)
    
    TABLE.index.name = 'Nsamples'
    
    if not (is_unique(df['name']) and is_unique(df['instantiator'])
            and is_unique(df['model_checking_type'])):
        print('Error: Benchmark instance not consistent, so cannot create table of lower bounds')
        assert False
        
    no_prop_list = np.sort(df['no_properties'].unique())
    Ns = np.sort(df['N'].unique())
    
    # For every number of samples and number of properties
    for N in Ns:
        for no_props in no_prop_list:
            
            # Get rows from dataframe
            rows = df.loc[(df['N'] == N) & (df['no_properties'] == no_props)]
            
            # Compute average scenario optimization run time and add to table
            average_time = np.mean(rows['scen_opt_time'])
            TABLE.loc[N, no_props] = average_time
    
            
elif args.mode == 'naive_comparison':
    # Comparison between scenario-based approach and naive Monte Carlo approach
    
    TABLE.index.name = 'Method'
    
    if not (is_unique(df['name']) and is_unique(df['instantiator'])
            and is_unique(df['model_checking_type'])):
        print('Error: Benchmark instance not consistent, so cannot create table of lower bounds')
        assert False
        
    rho = -1
    for b in json_bounds.values():
        if len(b) > 1:
            print('Error: Cannot create this table when multiple values of rho (cost of relaxation) are used')
            assert False
            
        if rho == -1:
            rho = list(b.keys())[0]
        else:
            if rho != list(b.keys())[0]:
                print('Error: Cannot create this table when the value of rho (costs of relaxation) differs between instances')
                assert False
                
    Ns = np.sort(df['N'].unique())
    
    # Loop over number of samples
    for N in Ns:
        
        # Initialize dictionaries
        bound_df = pd.DataFrame()
        naive_df = pd.DataFrame()
        
        rows = df.loc[df['N'] == N]
        
        # Loop over filtered set of rows
        for ID in rows['ID']:
            
            # Remove frequentist value for this table
            bounds_no_freq = json_bounds[ID][rho]
            bounds_no_freq.pop('frequentist')
            
            # Add bounds for both methods to the sub DataFrame
            bound_df = bound_df.append(bounds_no_freq, ignore_index=True)
            naive_df = naive_df.append(json_montecarlo[ID], ignore_index=True)
            
        cols = [str(col)+'_N='+str(N) for col in bound_df.columns]
        
        # Comptue average values
        bound_df = bound_df.mean()
        naive_df = naive_df.mean()
        
        # Add to main output table
        TABLE.loc['Our approach', cols] = bound_df.values
        TABLE.loc['Baseline', cols] = naive_df.values
           
        
root_dir = os.path.dirname(os.path.abspath(__file__))

outpath = path(root_dir, '', args.outfile)
outfolder = outpath.rsplit('/', 1)[0]

# Create output subfolder
Path(outfolder).mkdir(parents=True, exist_ok=True)
               
TABLE.to_csv(outpath, sep=';')
print('Final table:')
print(TABLE)
print('Table export (as csv) to:', args.outfile)