# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:15:51 2022

@author: Thom Badings
"""

import pickle
import pandas as pd

# Load pickled data
P = pickle.load(open('../red.dat',"rb"))

# Number of valuations
N = len(P)

# Define empty dictionary to store parameter valuations
dic = {'p11': [0]*N,
       'p22': [0]*N, 
       'p33': [0]*N, 
       'p44': [0]*N}

# For each DTMC
for i,p in P.items():

    # Load transition probabilities
    curr = p['mc_models'][0]    
    dic['p11'][i] = curr['P'][0,0]
    dic['p22'][i] = curr['P'][1,1]
    dic['p33'][i] = curr['P'][2,2]
    dic['p44'][i] = curr['P'][3,3]

# Convert to strings
dic['p11'] = str(dic['p11'])
dic['p22'] = str(dic['p22'])
dic['p33'] = str(dic['p33'])
dic['p44'] = str(dic['p44'])

# Export resulting Pandas Series to CSV file
SR = pd.Series(dic, name='values')
SR.index.name = 'name'
SR.to_csv('models/dtmc/sewer_pipe/parameters.csv', sep=';')