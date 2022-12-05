#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: glosat-iter-clusters.py
#------------------------------------------------------------------------------
#
# Version 0.1
# 5 December, 2022
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
import pickle
import csv
from datetime import datetime
import netCDF4
from netCDF4 import Dataset, num2date, date2num
import random

# System libraries:
import os

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16 

npowersof2 = 10
use_reproducible = True

df_temp_file = 'DATA/df_temp_qc.pkl'
	
#------------------------------------------------------------------------------
# LOAD: station archive dataframe 
#------------------------------------------------------------------------------
                          
df_temp = pd.read_pickle( df_temp_file, compression='bz2')  

# MASK: stations without lat or lon

stationcode_fails = df_temp[ (np.isnan( df_temp['stationlat'] )) | (np.isnan( df_temp['stationlon'] )) ].stationcode.unique()
df_temp_nonan = df_temp.drop( df_temp[ (np.isnan( df_temp['stationlat'] )) | (np.isnan( df_temp['stationlon'] )) ].index )
df_temp = df_temp_nonan.copy()

# MASK: stations only in US

df_usa = df_temp[ df_temp['stationcountry'] == 'USA' ]

stationcodes = df_usa.stationcode.unique()
stationlats = df_usa.groupby('stationcode').mean()['stationlat'].values
stationlons = df_usa.groupby('stationcode').mean()['stationlon'].values
stationidx = np.arange( len(stationcodes) ) 

# RANDOMLY SELECT N (POWER OF 2) STATIONS

for i in range( npowersof2 + 1 ):

    nstations = 2**i
        
    if use_reproducible == True:
        
        rng = np.random.default_rng(20221205) # seed for reproducibility
        idxlist = rng.choice( stationidx, nstations, replace=False )
        
    else:
        
        idxlist = np.random.choice( stationidx, nstations, replace=False )
                   
    # EXTRACT: cluster
    
    codes = stationcodes[ idxlist ]
    lats = stationlats[ idxlist ]
    lons = stationlons[ idxlist ]
    
    # SAVE: separate cluster files
    
    da = df_usa.copy()   
    df_temp_cluster = da[ da["stationcode"].isin( codes )].sort_values(['stationcode','year'], ascending=[True, True]).reset_index(drop=True)     
    clusterFile = 'df_temp_cluster_' + str(i).zfill(2) + '.pkl'
    df_temp_cluster.to_pickle( clusterFile, compression='bz2')
    
#------------------------------------------------------------------------------
print('** END')
