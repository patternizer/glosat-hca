#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: glosat-hca.py
#------------------------------------------------------------------------------
#
# Version 0.4
# 10 November, 2022
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

maxsize = 700
nclusters = 40

fontsize = 16 
export_pkl = True
plot_separate_clusters = True # ( default = True )

df_temp_file = 'DATA/df_temp_qc.pkl'
	
use_dark_theme = False
if use_dark_theme == True:
    default_color = 'white'
else:    
    default_color = 'black'    	
cmap = 'nipy_spectral'
 
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

"""
 think the kind of method needed would be hierarchical clustering - the sklearn package is this one:
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
and there are a range of methods implemented.

The distance matrix could be the output of prepare_dists in my glosat_homogenization module if you don't have your own already, 
then fed in as the argument to 'fit' using affinity="precomputed" in the constructor.
The only wrinkle is that the cluster closure condition is usually based on distance or number of clusters rather than cluster size. 
It will probably be necessary to use compute_full_tree and then recursively search the tree identifying nodes which contain fewer than 
the threshold number of stations but whose parents contain more than the threshold number.
"""

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
        
n = [ len( labels[labels==i] ) for i in range(len(np.unique(labels))) ]      
    
# SAVE: dataframe of clusters

dg = pd.DataFrame({'stationcode':stationcodes, 'lon':stationlons, 'lat':stationlats, 'cluster':labels})
dg.to_pickle('df_clusters.pkl', compression='bz2')
dc = list( dg.groupby('cluster')['stationcode'] )

# SAVE: separate cluster files

if export_pkl == False:
    
    # EXPORT: stationlist for each cluster to CSV
        
    for i in range(len(n)):
        
        cluster_stationcodes = dc[i][1].values        
        clusterFile = open( str(i).zfill(2)+'.csv', 'w' )
        wr = csv.writer( clusterFile, delimiter=',', lineterminator='\n' )
        for x in cluster_stationcodes : wr.writerow ([x])

else:
    
    for i in range(len(n)):
        
        cluster_stationcodes = dc[i][1].values            
        da = df_temp.copy()
        df_temp_cluster = da[ da["stationcode"].isin( cluster_stationcodes )]        
        clusterFile = 'df_temp_cluster_' + str(i).zfill(2) + '.pkl'
        df_temp_cluster.to_pickle( clusterFile, compression='bz2')
    
#==============================================================================
# PLOTS
#==============================================================================

# PLOT: global clusters ( all )
    
figstr = 'global-clusters-map' + '-' + str(nclusters).zfill(2) + '_' + str(len(dg.cluster.unique())).zfill(2) + '.png'
titlestr = 'HCA: clusters: 1st iteration=' + str(nclusters) + ', 2nd iteration=' + str(len(dg.cluster.unique())) + ' clusters'
colorbarstr = r'Cluster'

fig  = plt.figure(figsize=(15,10))
p = ccrs.PlateCarree(central_longitude=0)
ax = plt.axes(projection=p)
ax.set_global()
ax.set_extent([-180, 180, -90, 90], crs=p)    
gl = ax.gridlines(crs=p, draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='-')
gl.top_labels = False
gl.right_labels = False
gl.xlines = True
gl.ylines = True
gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': fontsize}
gl.ylabel_style = {'size': fontsize}              
plt.scatter( x = dg['lon'], y = dg['lat'], c = dg['cluster'], marker='s', s=1, alpha=0.5, transform=ccrs.PlateCarree(), cmap=cmap)  
ax.stock_img()
cb = plt.colorbar(shrink=0.7, extend='both')    
cb.set_label(colorbarstr, labelpad=10, fontsize=fontsize)
cb.ax.tick_params( labelsize=fontsize)
plt.title( titlestr, fontsize=fontsize )
plt.savefig( figstr, dpi=300, bbox_inches='tight')
plt.close('all')

if plot_separate_clusters == True:

    # PLOT: global clusters ( one per map )
    
    for i in range( len(n) ):
                
        x = dg[ dg['cluster']==i ]['lon']
        y = dg[ dg['cluster']==i ]['lat']
        c = np.ones(len(x))*i 
        
        figstr = 'global-clusters-map' + '-' + str(nclusters).zfill(2) + '_' + str(len(dg.cluster.unique())).zfill(2) + '_' + 'cluster' + '_' + str(i).zfill(2) + '.png'
        titlestr = 'HCA: clusters: 1st iteration=' + str(nclusters) + ', 2nd iteration=' + str(len(dg.cluster.unique())) + ': cluster=' + str(i).zfill(2)
        
        fig  = plt.figure(figsize=(15,10))
        p = ccrs.PlateCarree(central_longitude=0)
        ax = plt.axes(projection=p)
        ax.set_global()
        ax.set_extent([-180, 180, -90, 90], crs=p)    
        gl = ax.gridlines(crs=p, draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='-')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
        gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': fontsize}
        gl.ylabel_style = {'size': fontsize}              
        plt.scatter( x = x, y = y, c = c, marker='s', s=5, alpha=0.5, transform=ccrs.PlateCarree(), cmap=cmap)  
        ax.stock_img()
        plt.title( titlestr, fontsize=fontsize )
        plt.savefig( figstr, dpi=300, bbox_inches='tight')
        plt.close('all')

# PLOT: bar chart of cluster membership

figstr = 'global-clusters-membership' + '-' + str(nclusters).zfill(2) + '_' + str(len(dg.cluster.unique())).zfill(2) + '.png'
titlestr = 'HCA: cluster membership: 1st iteration=' + str(nclusters) + ', 2nd iteration=' + str(len(dg.cluster.unique())) + ' clusters'

fig,ax  = plt.subplots( figsize=(15,10) )
plt.bar( x=np.arange( 1, len(n)+1 ), height=n )
#plt.fill_between( np.arange( 1, len(n)+1 ), np.nanmedian(n) - 1.96*(np.nanpercentile(n,75)-np.nanpercentile(n,25)), np.nanmedian(n) + 1.96*(np.nanpercentile(n,75)-np.nanpercentile(n,25)), color='r', alpha=0.1, label=r'M$/pm$1.96*IQR' )
plt.fill_between( np.arange( 1, len(n)+1 ), np.nanpercentile( n, 25 ), np.nanpercentile( n, 75 ), color='r', alpha=0.2, label='IQR=' + str( int( np.nanpercentile(n,75) - np.nanpercentile(n,25) ) ) + ' stations/cluster' )
plt.hlines( xmin=1, xmax=len(n), y=np.nanmedian( n ),  colors='r', label='Median=' + str( int(np.nanmedian(n))) + ' stations/cluster' )
plt.xlim(0.5, len(n)+0.5)
plt.ylim(0, 700)
plt.legend(loc='upper left', fontsize=fontsize)
plt.title( titlestr, fontsize=fontsize)
plt.ylabel("N (stations)", fontsize=fontsize )
plt.xlabel("Cluster", fontsize=fontsize )
ax.tick_params(labelsize=fontsize)
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close('all')

# PLOT: dendrogram of linkage matrix Y for levels [1,5]

for i in range(1,6):

    nlevels=i
    
    figstr = 'global-clusters-dendrogram' + '-' + str(nlevels).zfill(2) + '.png'
    titlestr = 'HCA: dendrogram'
    
    fig,ax  = plt.subplots( figsize=(15,10) )
    dendrogram( Y, truncate_mode="level", p=nlevels )
    plt.title( "HCA: dendrogram ( nlevels=" + str(nlevels) + ' )', fontsize=fontsize)
    plt.xlabel("Number of stations per node (or index of station if no parenthesis) ", fontsize=fontsize )
    ax.tick_params(labelsize=fontsize)
    plt.savefig(figstr, dpi=300, bbox_inches='tight')
    plt.close('all')

#------------------------------------------------------------------------------
print('** END')
