#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: quickview.py
#------------------------------------------------------------------------------
# Version 0.1
# 23 May, 2022
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

reconstruct_clusters = False
base_start, base_end = 1961, 1990
stationname = []
stationcode = []
#stationname = 'CET'
#stationname = 'HORNSUND'
#stationcode = '085997'
stationcode = '037401'

#------------------------------------------------------------------------------
# LOAD: dataframe
#------------------------------------------------------------------------------

if reconstruct_clusters == True:

    nclusters = 40
    df_temp_reconstructed = pd.read_pickle('OUT/' + str(nclusters).zfill(2) + '/cluster-pkl/df_temp_reconstructed.pkl', compression='bz2')
    df_temp_reconstructed = df_temp_reconstructed.sort_values(['stationcode','year'], ascending=[True, True]).reset_index(drop=True)
    df_temp = df_temp_reconstructed.copy()

else:
	
    df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')

#------------------------------------------------------------------------------
# ARCHIVE: integrity
#------------------------------------------------------------------------------

stationcodes = df_temp['stationcode'].unique()
nstations = len(stationcodes)
stationlats = df_temp.groupby('stationcode').mean().stationlat.values
stationlons = df_temp.groupby('stationcode').mean().stationlon.values
stationyears = df_temp.groupby('stationcode').count().year.values
    
# FIND: stations with 0 years --> array([], dtype=object)

stations_nodata = stationcodes[ stationyears == 0 ]

# FIND: stations with missing lats
# --> array(['085997', '099999', '685807', '688607', '967811', '999099', '999216'], dtype=object)    

stations_nolats = stationcodes[ ~np.isfinite( stationlats ) ]

# FIND: stations with missing lons 
# --> array(['085997', '099999', '685807', '688607', '967811', '999096', '999099', '999216'], dtype=object)

stations_nolons = stationcodes[ ~np.isfinite( stationlons ) ]

print('N(stations)=', nstations)
print('stations (missing data)=', len(stations_nodata), ' :', stations_nodata)
print('stations (missing lat)=', len(stations_nolats), ' :',  stations_nolats)
print('stations (missing lon)=', len(stations_nolons), ' :',  stations_nolons)
  
#------------------------------------------------------------------------------
# COMPUTE: summary year stats
#------------------------------------------------------------------------------

print('YEAR: summary stats:', df_temp.describe().year )

#------------------------------------------------------------------------------
# SPOT CHECK: selected station
#------------------------------------------------------------------------------

# test station

if len(stationname) == 0:	
	stationname = df_temp[ df_temp['stationcode'] == stationcode ].stationname.unique()[0]
else:
	stationcode = df_temp[ df_temp['stationname'].str.contains( stationname, case=False) ].stationcode.unique()[0]
stationlat = df_temp[ df_temp['stationcode'] == stationcode ].stationlat.unique()[0]
stationlon = df_temp[ df_temp['stationcode'] == stationcode ].stationlon.unique()[0]
print('test station=', stationname)
print('test stationcode=', stationcode)
print('test stationlat=', stationlat)
print('test stationlon=', stationlon)
print( df_temp[ df_temp['stationname'].str.contains( stationname, case=False) ].describe().year )

#------------------------------------------------------------------------------
# COMPUTE: station anomaly series
#------------------------------------------------------------------------------

df_base = df_temp[ (df_temp.year>=base_start) & (df_temp.year<=base_end) ]
normals = df_base.groupby('stationcode').mean().iloc[:,1:13]
counts = df_base.groupby('stationcode').count().iloc[:,1:13]
normals[ counts <= 15 ] = np.nan

df_temp_station = df_temp[ df_temp.stationcode==stationcode ].reset_index(drop=True)
df_anom_station = df_temp_station.copy()
normals_station = normals[ normals.index==stationcode].reset_index(drop=True)
if np.isfinite(normals_station).values.sum() == 0:
    print('Station has no normals')
else:
    for i in range(1,13): 
        df_anom_station[str(i)] = df_temp_station[str(i)] - normals_station[str(i)][0]

    t_station = pd.date_range( start=str(df_anom_station.year.iloc[0]), end=str(df_anom_station.year.iloc[-1]+1), freq='MS')[0:-1]                                                                                                                                          
    ts_station = []    
    for i in range(len(df_anom_station)):            
        monthly = df_anom_station.iloc[i,1:13]
        ts_station = ts_station + monthly.to_list()    
    ts_station = np.array(ts_station)   

    fig, ax = plt.subplots()
    plt.plot(t_station, pd.Series(ts_station).rolling(24).mean())  
    plt.ylabel('2m Temperature Anomaly from ' + str(base_start) + '-' + str(base_end) )  
    plt.title(stationcode + ':' + stationname + ' (24m MA)')
    fig.savefig(stationname + '_quickplot.png', dpi=300)
    
#------------------------------------------------------------------------------
print('** END')

