# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:15:51 2022

@author: Thom Badings
"""

import pickle
import pandas as pd

# Load pickled data
P = pickle.load(open('../blue.dat',"rb"))

# Number of valuations
N = len(P)

# Define empty dictionary to store parameter valuations
dic = {'p11': [0]*N,
       'p22': [0]*N, 
       'p33': [0]*N, 
       'p44': [0]*N,
       's1_init': [0]*N,
       's2_init': [0]*N,
       's3_init': [0]*N,
       's4_init': [0]*N,
       's5_init': [0]*N}

# For each DTMC
for i,p in P.items():

    # Load transition probabilities
    curr = p['mc_models'][0]    
    dic['p11'][i] = curr['P'][0,0]
    dic['p22'][i] = curr['P'][1,1]
    dic['p33'][i] = curr['P'][2,2]
    dic['p44'][i] = curr['P'][3,3]
    
    dic['s1_init'][i] = curr['p0'][0]
    dic['s2_init'][i] = curr['p0'][1]
    dic['s3_init'][i] = curr['p0'][2]
    dic['s4_init'][i] = curr['p0'][3]
    dic['s5_init'][i] = curr['p0'][4]

# Convert to strings
dic['p11'] = str(dic['p11'])
dic['p22'] = str(dic['p22'])
dic['p33'] = str(dic['p33'])
dic['p44'] = str(dic['p44'])

dic['s1_init'] = str(dic['s1_init'])
dic['s2_init'] = str(dic['s2_init'])
dic['s3_init'] = str(dic['s3_init'])
dic['s4_init'] = str(dic['s4_init'])
dic['s5_init'] = str(dic['s5_init'])

# Export resulting Pandas Series to CSV file
SR = pd.Series(dic, name='values')
SR.index.name = 'name'
SR.to_csv('models/dtmc/sewer_pipe/parameters.csv', sep=';')