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
import os

# Parse argument that gives the file to read from
parser = argparse.ArgumentParser(description="Generate average lower bounds table")
parser.add_argument('--file', type=str, action="store", dest='file', 
                    default=None, help="File to load and convert")
parser.add_argument('--outfile', type=str, action="store", dest='outfile', 
                    default=None, help="Name of file to export table to")
parser.add_argument('--no_props', type=str, action="store", dest='no_props', 
                    default=None, help="Number of properties for row to add")

args = parser.parse_args()    

# Read CSV file
df = pd.read_csv(args.file, sep=';')
df = df.drop(columns=['#'])

# Compute average run times for current number of properties
avg = df.mean(axis=0)

# Convert to dataframe
df_avg = pd.DataFrame(columns = df.columns)
df_avg.loc[int(args.no_props)] = np.round(avg, 2)
df_avg.index.name = 'no_props'

# If the output file already exists, append to it
if os.path.exists(args.outfile):
    df_avg_existing = pd.read_csv(args.outfile, sep=';', index_col='no_props')
    df_avg = pd.concat([df_avg_existing, df_avg])
    
# Export to file
df_avg.to_csv(args.outfile, sep=';')