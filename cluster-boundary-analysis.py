#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: cluster-boundary-analysis.py
#------------------------------------------------------------------------------
#
# Version 0.1
# 18 December, 2022
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

# System libraries:
import os
import glob

# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns; sns.set()
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.ticker as mticker

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16 

cluster_halo_path = 'run-glosat-p04c-ebc-halo/lek-cluster-pkl/'
cluster_no_halo_path = 'run-glosat-p04c-ebc-no-halo/lek-cluster-pkl/'

#------------------------------------------------------------------------------
# LOAD: clusters 
#------------------------------------------------------------------------------

cluster_halo_pkllist = glob.glob( cluster_halo_path + "*.pkl" )
cluster_no_halo_pkllist = glob.glob( cluster_no_halo_path + "*.pkl" )

nclusters = len( cluster_no_halo_pkllist )
rms_e = []
rms_n = []

for i in range( nclusters ):
   
    df_temp_expect_halo = pd.read_pickle( cluster_halo_pkllist[i], compression='bz2' )
    df_temp_expect_no_halo = pd.read_pickle( cluster_no_halo_pkllist[i], compression='bz2' )
        
    # FIND: common stationcodes
    
    stationcode_halo = df_temp_expect_halo.stationcode.unique()
    stationcode_no_halo = df_temp_expect_no_halo.stationcode.unique()
    
    common = np.intersect1d( stationcode_halo, stationcode_no_halo )
    
    # EXTRACT: stations not in halo
    
    df_temp_expect = df_temp_expect_halo[ df_temp_expect_halo.stationcode.isin(common) ].reset_index(drop=True)
    
    for j in range(12):
        
        rms_e_month = np.sqrt( np.nanmean( ( np.array(df_temp_expect_no_halo['e'+str(j+1)]) - np.array(df_temp_expect['e'+str(j+1)]) )**2.0 ) ) 
        rms_e.append( rms_e_month )

        rms_n_month = np.sqrt( np.nanmean( ( np.array(df_temp_expect_no_halo['n'+str(j+1)]) - np.array(df_temp_expect['n'+str(j+1)]) )**2.0 ) ) 
        rms_n.append( rms_n_month )

rms_e_array = np.reshape( rms_e, [ nclusters, 12 ] )
rms_n_array = np.reshape( rms_n, [ nclusters, 12 ] )

rms_e_vec = np.nanmean( rms_e_array, axis=1 )
rms_n_vec = np.nanmean( rms_n_array, axis=1 )
        
#==============================================================================
# PLOTS
#==============================================================================

ncolors = 20
cmap = cm.get_cmap('PiYG', ncolors ) # discrete colors

# PLOT: heatmap of local expectation RMS

rms_e_mean = np.nanmean( rms_e_array )
rms_e_sd = np.nanstd( rms_e_array )
vmin = rms_e_mean - rms_e_sd
vmax = rms_e_mean + rms_e_sd

figstr = 'cluster-rms-e.png'
titlestr = 'HCA: local expectation RMS (without-with halo): N clusters=' + str(nclusters)

fig,ax  = plt.subplots( figsize=(15,10) )
sns.heatmap( rms_e_array, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'drawedges': True, 'shrink':0.7, 'extend':'both', 'label':'RMS (' + r'$\mu=$' + str(np.round(rms_e_mean,3)) + r' $\sigma=$' + str(np.round(rms_e_sd,3)) + ')' + r' $^{\circ}C$'})
ax.set_xticks(np.arange(12)+0.5)
ax.set_xticklabels(np.arange(1,13), rotation='horizontal')
ax.set_yticks(np.arange(nclusters)+0.5)
ax.set_yticklabels(np.arange(1,nclusters+1), rotation='horizontal')
plt.xlabel("Month", fontsize=fontsize )
plt.ylabel("Cluster", fontsize=fontsize )
plt.title( titlestr, fontsize=fontsize)
ax.tick_params(labelsize=8)
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.close('all')

# PLOT: heatmap of normals RMS

rms_n_mean = np.nanmean( rms_n_array )
rms_n_sd = np.nanstd( rms_n_array )
vmin = rms_n_mean - rms_n_sd
vmax = rms_n_mean + rms_n_sd
                 
figstr = 'cluster-rms-n.png'
titlestr = 'HCA: normal RMS (without-with halo): N clusters=' + str(nclusters)

fig,ax  = plt.subplots( figsize=(15,10) )
sns.heatmap( rms_n_array, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'drawedges': True, 'shrink':0.7, 'extend':'both', 'label':'RMS (' + r'$\mu=$' + str(np.round(rms_n_mean,3)) + r' $\sigma=$' + str(np.round(rms_n_sd,3)) + ')' + r' $^{\circ}C$'})
ax.set_xticks(np.arange(12)+0.5)
ax.set_xticklabels(np.arange(1,13), rotation='horizontal')
ax.set_yticks(np.arange(nclusters)+0.5)
ax.set_yticklabels(np.arange(1,nclusters+1), rotation='horizontal')
plt.xlabel("Month", fontsize=fontsize )
plt.ylabel("Cluster", fontsize=fontsize )
plt.title( titlestr, fontsize=fontsize)
ax.tick_params(labelsize=8)
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.close('all')

# PLOT: cluster mean RMS

figstr = 'cluster-rms-mean.png'
titlestr = 'HCA: cluster mean RMS (without-with halo): N clusters=' + str(nclusters)
                        
fig, ax = plt.subplots(figsize=(15,10))     
plt.plot( np.arange(1,nclusters+1), rms_e_vec, marker='o', ls='-', lw=0.5, color='red', label='Expectations' + r' ($\mu=$' + str( np.round( np.nanmean( rms_e_vec ), 3 ) ) + r' $\sigma=$' + str( np.round( np.nanstd( rms_e_vec ), 3 ) ) + ')' )
plt.plot( np.arange(1,nclusters+1), rms_n_vec, marker='o', ls='-', lw=0.5, color='green', label='Normals' + r' ($\mu=$' + str( np.round( np.nanmean( rms_n_vec ), 3 ) ) + r' $\sigma=$' + str( np.round( np.nanstd( rms_n_vec ), 3 ) ) + ')' )
plt.axhline( y=np.nanmean( rms_e_vec ), ls='dashed', lw=1, color='red' )
plt.axhline( y=np.nanmean( rms_n_vec ), ls='dashed', lw=1, color='green' )
plt.xlim(0,nclusters+1)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='upper left', markerscale=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
plt.xlabel('Cluster', fontsize=fontsize)
plt.ylabel('RMS, '+ r'$^{\circ}C$', fontsize=fontsize)
plt.tick_params(labelsize=fontsize, colors='green')    
plt.title(titlestr, fontsize=fontsize, pad=10)
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------
print('** END')
