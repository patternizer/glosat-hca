#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: glosat-hca-halo-robust.py
#------------------------------------------------------------------------------
#
# Version 0.1
# 13 December, 2022
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import numpy as np
from numpy import ma
import pandas as pd
import xarray as xr
import pickle
import csv
from datetime import datetime
import netCDF4
from netCDF4 import Dataset, num2date, date2num

# System libraries:
import os

# ML libraries:

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree, cut_tree

# Matplotlib libraries:
import matplotlib    
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker   # for gridlines
import matplotlib.cm as cm            # for cmap
import matplotlib.patches as mpatches # for polygons
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Cartopy libraries:
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
#from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE
#shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')

# Seaborn libraries
import seaborn as sns; sns.set()

projection = 'platecarree'

if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(); threshold = 0
if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

maxsize = 700
nclusters = 40

fontsize = 16 
plot_separate_clusters = True # ( default = True )

df_temp_file = 'DATA/df_temp_qc.pkl'
	    
#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def prepare_dists(lats, lons):
    """
    Prepare distance matrix from vectors of lat/lon in degrees assuming
    spherical earth
  
    Parameters:
        lats (vector of float): latitudes
        lons (vector of float): longitudes
  
    Returns:
        (matrix of float): distance matrix in km
    """
    las = np.radians(lats)
    lns = np.radians(lons)
    dists = np.zeros([las.size,las.size])
    for i in range(lns.size):
        dists[i,:] = 6371.0*np.arccos( np.minimum( (np.sin(las[i])*np.sin(las) + np.cos(las[i])*np.cos(las)*np.cos(lns[i]-lns) ), 1.0 ) )
    return dists

def get_time():
    
    # Calculate current time

    now = datetime.now()
    currentdy = str(now.day).zfill(2)
    currentmn = str(now.month).zfill(2)
    currentyr = str(now.year)
    titletime = str(currentdy) + '/' + currentmn + '/' + currentyr    
    
    return titletime

#------------------------------------------------------------------------------
# LOAD: station coords 
#------------------------------------------------------------------------------
                          
df_temp = pd.read_pickle( df_temp_file, compression='bz2')  

#------------------------------------------------------------------------------
# MASK: stations without lat or lon
#------------------------------------------------------------------------------

stationcode_fails = df_temp[ (np.isnan( df_temp['stationlat'] )) | (np.isnan( df_temp['stationlon'] )) ].stationcode.unique()
df_temp_nonan = df_temp.drop( df_temp[ (np.isnan( df_temp['stationlat'] )) | (np.isnan( df_temp['stationlon'] )) ].index )
df_temp = df_temp_nonan.copy()

stationcodes = df_temp.stationcode.unique()
stationlats = df_temp.groupby('stationcode').mean()['stationlat'].values
stationlons = df_temp.groupby('stationcode').mean()['stationlon'].values

#------------------------------------------------------------------------------
# COMPUTE: distance matrix
#------------------------------------------------------------------------------

X = prepare_dists( stationlats, stationlons )
X = np.nan_to_num( X )

#------------------------------------------------------------------------------
# COMPUTE: HCA --> clustering
#------------------------------------------------------------------------------

model = AgglomerativeClustering( n_clusters = nclusters, affinity="precomputed", linkage='complete', distance_threshold=None, compute_distances=True, compute_full_tree=True ).fit( X )
labels = model.labels_

# COUNT:  number of stations in each cluster

n = [ len( labels[ labels == i ] ) for i in range( nclusters ) ]      

# FIND: clusters > maxstations

labelfaillist = np.arange( nclusters )[ np.array( n ) > maxsize ]

# ITERATE: until nmembers < maxstations

for label in labelfaillist:
    
    codes = stationcodes[ labels == label ]
    lats = stationlats[ labels == label ]
    lons = stationlons[ labels == label ]
    X_subset = prepare_dists( lats, lons )
    X_subset = np.nan_to_num( X_subset )

    idx_label = np.arange( len( labels ) )[ labels == label ]
    n_subset = n[ label ]

    nclusters_subset = 2    
    while (np.array( n_subset ) > maxsize).sum() > 0:

        model_subset = AgglomerativeClustering( n_clusters = nclusters_subset, affinity="precomputed", linkage='complete', distance_threshold=None, compute_distances=True, compute_full_tree=True ).fit( X_subset )
        labels_subset = model_subset.labels_   
        n_subset = [ len( labels_subset[labels_subset==i] ) for i in range( nclusters_subset ) ]      

        if (np.array(n_subset) > maxsize).sum() > 0:

            nclusters_subset += 1

        else: 

            for k in range( len( n_subset ) ):
               
                label_subset_mask = labels_subset == k   
                idx_label_subset = idx_label[ label_subset_mask ]                
                labels[ idx_label_subset ] = k + (label*1000)

            break                
        
# MAP: cluster labels onto integer series

for i in range(len(np.unique(labels))):
    
    idx = labels == np.unique(labels)[i]
    labels [ idx ] = i        
      
#------------------------------------------------------------------------------
# CONSTRUCT: dataframe of clusters
#------------------------------------------------------------------------------

df = pd.DataFrame({'stationcode':stationcodes, 'lon':stationlons, 'lat':stationlats, 'cluster':labels})
dc = list( df.groupby('cluster')['stationcode'] )
    
#------------------------------------------------------------------------------
# EXTRACT: (loop) cluster and halo dataframes
#------------------------------------------------------------------------------

nstations = len(df)
nclusters = len(df.cluster.unique())

for i in range(nclusters):
    
    # EXTRACT: cluster dataframe
    
    df_cluster = df[ df.cluster == i ].reset_index(drop=True)    
    
    # LIST: stations in cluster
    
    stationcodes_cluster = df_cluster.stationcode.values
        
    # BOOLEAN MASK:
        
    boolean_mask = np.array( nstations * [False] )

    for j in range( nstations ):
        
        for k in range( len(stationcodes_cluster) ):
            
            idx = stationcodes_cluster[k] == stationcodes[j]        

            if idx == True:
                
                boolean_mask[j] = True
        
    # EXTRACT: distance matrix columns for cluster and calculate minimum distances to cluster stations

    distance_matrix_rows = X[:,boolean_mask]
    distance_matrix_rows_min = np.min( distance_matrix_rows, axis=1 ) 

    # EXTRACT: halo stations from distance matrix sorted by distance

    dg = df.copy().reset_index(drop=True)
    dg['distance'] = distance_matrix_rows_min
    dg_sorted = dg.sort_values('distance')
    dg_external = dg_sorted[ dg_sorted.distance > 1 ].reset_index(drop=True)

    nstations_cluster = stationcodes_cluster.shape[0]
    nstations_halo = int(nstations_cluster/2)

    df_halo = dg_external.iloc[0:nstations_halo,:]

    stationcodes_halo = df_halo.stationcode.values

    stationcodes_cluster_halo = np.array( list(stationcodes_cluster) + list(stationcodes_halo) )
    
    # SAVE: dataframe of clusters and dataframe of clusters + halos
                    
    da = df_temp.copy()
    df_temp_cluster_halo = da[ da["stationcode"].isin( stationcodes_cluster_halo )]        
    df_temp_cluster = da[ da["stationcode"].isin( stationcodes_cluster )]        
    cluster_File = 'df_temp_cluster_' + str(i).zfill(2) + '.pkl'
    cluster_halo_File = 'df_temp_cluster_halo_' + str(i).zfill(2) + '.pkl'
    df_temp_cluster.to_pickle( cluster_File, compression='bz2')
    df_temp_cluster_halo.to_pickle( cluster_halo_File, compression='bz2')
    
    #==============================================================================
    # PLOT: cluster + halo
    #==============================================================================

    #cmap = cm.get_cmap('nipy_spectral', len(np.unique(labels)) ) # discrete colors
    cmap = cm.get_cmap('PiYG', len(np.unique(labels)) ) # discrete colors
    
    figstr = 'global-cluster-halo' + '-' + str(i).zfill(2) + '.png'
    titlestr = 'HCA: cluster ' + str(i+1).zfill(2) + ' + halo (n clusters=' + str( nclusters ) + ')'
                                                                                
    fig, ax = plt.subplots(figsize=(15,10), subplot_kw=dict(projection=p))
    ax.stock_img()
    #ax.add_feature(cf.COASTLINE, lw=2)

    ax.set_global()
    ax.gridlines()
    ax.set_extent([-180, 180, -90, 90], crs=p)    
                                   
    gl = ax.gridlines(crs=p, draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True
    gl.xlocator = mticker.FixedLocator([-120,-60,0,60,120])
    gl.ylocator = mticker.FixedLocator([-60,-30,0,30,60])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize}              
    
    plt.scatter( x = df['lon'], y = df['lat'], marker='o', s=3, facecolors='none', edgecolors='grey', lw=0.5, alpha=1, transform=p, cmap=cmap, label='External: n=' + str( nstations - nstations_cluster - nstations_halo) )      
    plt.scatter( x = df_cluster['lon'], y = df_cluster['lat'], marker='o', s=3, facecolors='none', edgecolors='blue', lw=0.5, alpha=1, transform=p, cmap=cmap, label='Cluster: n=' + str(nstations_cluster) )      
    plt.scatter( x = df_halo['lon'], y = df_halo['lat'], marker='o', s=3, facecolors='none', edgecolors='red', lw=0.5, alpha=1, transform=p, cmap=cmap, label='Halo: n=' + str(nstations_halo) )      

    plt.legend(loc='lower left', markerscale=3, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)        
    plt.title( titlestr, fontsize=fontsize )
    plt.savefig( figstr, dpi=600, bbox_inches='tight')
    plt.close('all')

#------------------------------------------------------------------------------
print('** END')
