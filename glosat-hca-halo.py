#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: glosat-hca-halo.py
#------------------------------------------------------------------------------
#
# Version 0.1
# 9 December, 2022
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

maxsize = 1024
nclusters = 30

fontsize = 16 
plot_separate_clusters = True # ( default = True )

df_temp_file = 'DATA/df_temp_qc.pkl'
	
use_dark_theme = False
if use_dark_theme == True:
    default_color = 'white'
else:    
    default_color = 'black'    	
 
#----------------------------------------------------------------------------
# DARK THEME
#----------------------------------------------------------------------------

if use_dark_theme == True:
    
    matplotlib.rcParams['text.usetex'] = False
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Avant Garde', 'Lucida Grande', 'Verdana', 'DejaVu Sans' ]
    plt.rc('text',color='white')
    plt.rc('lines',color='white')
    plt.rc('patch',edgecolor='white')
    plt.rc('grid',color='lightgray')
    plt.rc('xtick',color='white')
    plt.rc('ytick',color='white')
    plt.rc('axes',labelcolor='white')
    plt.rc('axes',facecolor='black')
    plt.rc('axes',edgecolor='lightgray')
    plt.rc('figure',facecolor='black')
    plt.rc('figure',edgecolor='black')
    plt.rc('savefig',edgecolor='black')
    plt.rc('savefig',facecolor='black')
    
else:

    print('Using Seaborn graphics ...')
    
# Calculate current time

now = datetime.now()
currentdy = str(now.day).zfill(2)
currentmn = str(now.month).zfill(2)
currentyr = str(now.year)
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr    

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

def compute_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    
    return linkage_matrix
    
#------------------------------------------------------------------------------
# LOAD: station coords 
#------------------------------------------------------------------------------
                          
df_temp = pd.read_pickle( df_temp_file, compression='bz2')  

# MASK: stations without lat or lon

stationcode_fails = df_temp[ (np.isnan( df_temp['stationlat'] )) | (np.isnan( df_temp['stationlon'] )) ].stationcode.unique()
df_temp_nonan = df_temp.drop( df_temp[ (np.isnan( df_temp['stationlat'] )) | (np.isnan( df_temp['stationlon'] )) ].index )
df_temp = df_temp_nonan.copy()

stationcodes = df_temp.stationcode.unique()
stationlats = df_temp.groupby('stationcode').mean()['stationlat'].values
stationlons = df_temp.groupby('stationcode').mean()['stationlon'].values

# COMPUTE: distance matrix

X = prepare_dists( stationlats, stationlons )
X = np.nan_to_num( X )

# setting distance_threshold=0 ensures we compute the full tree
model = AgglomerativeClustering( n_clusters = nclusters, affinity="precomputed", linkage='complete', distance_threshold=None, compute_distances=True, compute_full_tree=True ).fit( X )
labels = model.labels_

# COMPUTE: linkage matrix (Y) for up to 5 levels of the dendrogram

Y = compute_dendrogram( model, truncate_mode="level", p=5 )
#clusters = cut_tree(linkage_matrix, n_clusters=nclusters)

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

# COMPUTE: number of members per cluster
        
#n = [ len( labels[labels==i] ) for i in range(len(np.unique(labels))) ]      
    
# SAVE: dataframe of clusters

dg = pd.DataFrame({'stationcode':stationcodes, 'lon':stationlons, 'lat':stationlats, 'cluster':labels})
dg.to_pickle('df_clusters.pkl', compression='bz2')
dc = list( dg.groupby('cluster')['stationcode'] )


# COMPUTE: cluster centroids

dg_centroids = dg.groupby('cluster').mean()
lon_centroids = dg_centroids.iloc[:,0].values
lat_centroids = dg_centroids.iloc[:,1].values
    
# COMPUTE: cluster outer radii

outer_station_deltas = []

for i in range(len(dg_centroids)):
    
    # COMPUTE: distance matrix from cluster centroid

    lon_cluster = dg[dg.cluster==i].lon
    lat_cluster = dg[dg.cluster==i].lat

    centroid_and_lon_cluster = np.hstack( [np.array(lon_centroids[i]), lon_cluster] )
    centroid_and_lat_cluster = np.hstack( [np.array(lat_centroids[i]), lat_cluster] )
    
    X = prepare_dists( centroid_and_lat_cluster, centroid_and_lon_cluster )
    X = np.nan_to_num( X )

#    outer_station_idx = np.argmax(X[0,:]) - 1
    outer_station_idx = np.arange( len(X) )[ X[0,:] == np.quantile(X[0,:], 0.5, interpolation='nearest') ][0] - 1
    outer_station_dist = X[0,outer_station_idx+1]

    outer_station_lon_delta = np.abs( lon_cluster - lon_centroids[i] )
    outer_station_lat_delta = np.abs( lat_cluster - lat_centroids[i] )
    outer_station_delta = np.max( [outer_station_lon_delta, outer_station_lat_delta] )

    if outer_station_delta > 60:
        
        print(i, outer_station_delta)
        outer_station_delta = 0

    outer_station_deltas.append( outer_station_delta )

dg_centroids['delta'] = np.array(outer_station_deltas)

'''

# SAVE: dataframe of clusters + halos

# SAVE: separate cluster files
    
for i in range(len(n)):
        
    cluster_stationcodes = dc[i][1].values            
    da = df_temp.copy()
    df_temp_cluster = da[ da["stationcode"].isin( cluster_stationcodes )]        
    clusterFile = 'df_temp_cluster_' + str(i).zfill(2) + '.pkl'
    df_temp_cluster.to_pickle( clusterFile, compression='bz2')

'''
    
#==============================================================================
# PLOTS
#==============================================================================

#cmap = cm.get_cmap('nipy_spectral', len(np.unique(labels)) ) # discrete colors
cmap = cm.get_cmap('PiYG', len(np.unique(labels)) ) # discrete colors
    

# PLOT: clusters + centroid + outer radius (no halos)

figstr = 'global-clusters-map-no-halos' + '-' + str(nclusters).zfill(2) + '_' + str(len(dg.cluster.unique())).zfill(2) + '.png'
titlestr = 'HCA: clusters + centroids + outer radii (n clusters=' + str(len(dg.cluster.unique())) + ')'
colorbarstr = r'Cluster'
                                                                                
fig, ax = plt.subplots(figsize=(15,10), subplot_kw=dict(projection=p))
#ax.stock_img()
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

for i in range(len(dg_centroids)):

    x = dg_centroids.lon[i]
    y = dg_centroids.lat[i]
    r = dg_centroids.delta[i] # degrees

    rgba = cmap(i/len(dg_centroids)) 
    poly = mpatches.Circle((x,y),r, color=rgba, transform=p)
    ax.add_patch(poly).set_alpha(0.5)

    if dg_centroids.delta[i] > 0:
        plt.scatter( x = dg_centroids.lon[i], y = dg_centroids.lat[i], c = 'black', marker='x', s=50, alpha=1, zorder=3, transform=p, cmap=cmap)      
    else:
        plt.scatter( x = dg_centroids.lon[i], y = dg_centroids.lat[i], c = 'red', marker='x', s=50, alpha=1, zorder=3, transform=p, cmap=cmap)      

plt.scatter( x = dg['lon'], y = dg['lat'], c = dg['cluster'], marker='o', s=3, alpha=1, transform=p, cmap=cmap)      

cb = plt.colorbar(shrink=0.7, extend='both')    
cb.set_label(colorbarstr, labelpad=10, fontsize=fontsize)
cb.ax.tick_params( labelsize=fontsize)
plt.title( titlestr, fontsize=fontsize )
plt.savefig( figstr, dpi=300, bbox_inches='tight')
plt.close('all')

# PLOT: centroid + outer radius + halos

figstr = 'global-clusters-map-halos' + '-' + str(nclusters).zfill(2) + '_' + str(len(dg.cluster.unique())).zfill(2) + '.png'
titlestr = 'HCA: outer radii + haloes (n clusters=' + str(len(dg.cluster.unique())) + ')'
colorbarstr = r'Cluster'
                                                                                
fig, ax = plt.subplots(figsize=(15,10), subplot_kw=dict(projection=p))
#ax.stock_img()
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

for i in range(len(dg_centroids)):

    x = dg_centroids.lon[i]
    y = dg_centroids.lat[i]
    r = dg_centroids.delta[i] # degrees

    poly = mpatches.Circle((x,y),1.5*r, color='k', transform=p)
    ax.add_patch(poly).set_alpha(0.5)

    rgba = cmap(i/len(dg_centroids)) 
    poly = mpatches.Circle((x,y),r, color=rgba, transform=p)
    ax.add_patch(poly).set_alpha(0.5)

    if dg_centroids.delta[i] > 0:
        plt.scatter( x = dg_centroids.lon[i], y = dg_centroids.lat[i], c = 'black', marker='x', s=50, alpha=1, zorder=3, transform=p, cmap=cmap)      
    else:
        plt.scatter( x = dg_centroids.lon[i], y = dg_centroids.lat[i], c = 'red', marker='x', s=50, alpha=1, zorder=3, transform=p, cmap=cmap)      

plt.scatter( x = dg['lon'], y = dg['lat'], c = dg['cluster'], marker='o', s=3, alpha=1, transform=p, cmap=cmap) 

cb = plt.colorbar(shrink=0.7, extend='both')    
cb.set_label(colorbarstr, labelpad=10, fontsize=fontsize)
cb.ax.tick_params( labelsize=fontsize)

plt.title( titlestr, fontsize=fontsize )
plt.savefig( figstr, dpi=300, bbox_inches='tight')
plt.close('all')


#------------------------------------------------------------------------------
print('** END')
