#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: stations_nonormals_and_normals_evolution.py
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
# Dataframe libraries:
import numpy as np
import numpy.ma as ma
import pandas as pd
import pickle
from datetime import datetime
import nc_time_axis
import cftime

# Stats libraries:
from scipy.interpolate import griddata
from scipy import spatial

# Plotting libraries:
import seaborn as sns; sns.set()
import matplotlib as mpl
import matplotlib.pyplot as plt; plt.close('all')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter, LatitudeLocator)
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 16

window = 10 # decadal
use_glosat_start = False # ( default = True ) False --> use edge of Pandas = 1678
if use_glosat_start == True:
    tstart, tend = 1781, 2022
else:
    tstart, tend = 1678, 2022
edges = np.arange( tstart, tend, step=window)
    
plot_nonormals_stations_world = False
plot_nonormals_stations_gridcount_world = False

show_gridlines = True
projection = 'mollweide'        

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def compute_gridded_counts( ds_lon, ds_lat ):

    step = 5
    grid_lat = np.arange( -90+step/2, 90+step/2, step )
    grid_lon = np.arange( -180+step/2, 180+step/2, step )
    X,Y = np.meshgrid( grid_lon, grid_lat)
    N = len(grid_lon) * len(grid_lat)
    x = X.reshape(N)
    y = Y.reshape(N)
    
    df = pd.DataFrame({'stationcode':ds_lon.index, 'stationlon':ds_lon, 'stationlat':ds_lat}).reset_index(drop=True)
    dg = pd.DataFrame({'lon':x, 'lat':y}, index=range(N))
    A = list(zip(*map(dg.get, ['lat', 'lon'])))
    tree = spatial.KDTree(A)
    lat = []
    lon = []
    for i in range(len(df)):
        pti = [df.loc[i]['stationlat'],df.loc[i]['stationlon']]        
        distancei,indexi = tree.query(pti)    
        lati = dg.loc[indexi,:]['lat']
        loni = dg.loc[indexi,:]['lon']        
        lat.append(lati)
        lon.append(loni)
    df['lon']=lon
    df['lat']=lat
    counts = df.groupby(['lon','lat']).count()['stationcode'].values
    lons = [ df.groupby(['lon','lat']).count().index[i][0] for i in range(len(counts)) ]
    lats = [ df.groupby(['lon','lat']).count().index[i][1] for i in range(len(counts)) ]
    
    dm = pd.DataFrame({'lats':lats, 'lons':lons, 'counts':counts})
    
    return dm

#------------------------------------------------------------------------------
# LOAD: absolute temperatures and normals
#------------------------------------------------------------------------------

df_temp_in = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')
df_anom_in = pd.read_pickle('DATA/df_anom.pkl', compression='bz2')
df_normals = pd.read_pickle('DATA/df_normals.pkl', compression='bz2')
df_temp = df_temp_in.copy()
df_anom = df_anom_in[df_anom_in['stationcode'].isin(df_normals[df_normals['sourcecode']>1]['stationcode'])]

dt = df_temp.copy() # all stations
da = df_anom.copy() # anomalies
ds = dt.copy()    
ds = ds.drop(da.index)
ds = ds.dropna()

# COMPUTE: number of short-segment series per time window

n_all = []
n_normals = []
n_nonormals = []
for i in range(len(edges)-1):
    
    t_start = edges[i]
    t_end = edges[i+1]-1

    dt_n = dt[ (dt.year>=t_start) & (dt.year<=t_end) ].stationcode.unique().shape[0]
    da_n = da[ (da.year>=t_start) & (da.year<=t_end) ].stationcode.unique().shape[0]
    ds_n = ds[ (ds.year>=t_start) & (ds.year<=t_end) ].stationcode.unique().shape[0]
    
    n_all.append( dt_n )
    n_normals.append( da_n )
    n_nonormals.append( ds_n )

n_nonormals_proportion = np.array( n_nonormals ) / np.array( n_normals )
    
#==============================================================================
# PLOTS
#==============================================================================

# PLOT: temporal evolution of short-segment series ( gridded 5x5 )

for i in range(len(edges)-1):

    t_start = edges[i]
    t_end = edges[i+1]-1
    
    dt_lon = dt[ (dt.year>=t_start) & (dt.year<=t_end) ].groupby('stationcode').mean()['stationlon']
    dt_lat = dt[ (dt.year>=t_start) & (dt.year<=t_end) ].groupby('stationcode').mean()['stationlat']
    da_lon = da[ (da.year>=t_start) & (da.year<=t_end) ].groupby('stationcode').mean()['stationlon']
    da_lat = da[ (da.year>=t_start) & (da.year<=t_end) ].groupby('stationcode').mean()['stationlat']    
    ds_lon = ds[ (ds.year>=t_start) & (ds.year<=t_end) ].groupby('stationcode').mean()['stationlon']
    ds_lat = ds[ (ds.year>=t_start) & (ds.year<=t_end) ].groupby('stationcode').mean()['stationlat']    
   
    if plot_nonormals_stations_gridcount_world == True:

        #------------------------------------------------------------------------------
        # PLOT: nonormals stations gridded counts on world map
        #------------------------------------------------------------------------------

        dm = compute_gridded_counts( ds_lon, ds_lat )
        
        x_list = dm.lats 
        y_list = dm.lons
        z_list = dm.counts
        length = np.size(x_list)
        N_x = np.unique(x_list)
        N_y = np.unique(y_list)
        X, Y = np.meshgrid(N_x,N_y)
        length_x = np.size(N_x)
        length_y = np.size(N_y)
        Z = np.full((length_x, length_y), np.nan)

        def f(x, y):
            for i in range(0, length):
                if (x_list[i] == x) and (y_list[i] == y):
                    return z_list[i]
                
        for i in range(0, length_x - 1):
            for j in range(0, length_y - 1):
                Z[i,j] = f(N_x[i], N_y[j])
                
        print('plot_nonormals_station_gridcell_counts ...')
            
        figstr = 'short-segment-stations-gridded-counts-' + str(t_start) + '-' + str(t_end) + '.png'
        titlestr = 'GloSAT.p04: stations without normals counts (gridded 5x5): ' + str(t_start) + '-' + str(t_end)
        
        fig  = plt.figure(figsize=(15,10))
        if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
        if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
        if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
        if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
        if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
        if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
        if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
        if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(central_longitude=-30); threshold = 0
        if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
        if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0
        if projection == 'winkeltripel': p = ccrs.WinkelTripel(central_longitude=0); threshold = 0
        ax = plt.axes(projection=p)    
        ax.set_global()
        ax.stock_img()                
        if show_gridlines == True:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='grey', alpha=0.5, linestyle='-')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.left_labels = False
            gl.right_labels = False
            gl.xlines = True
            gl.ylines = True
            gl.xlocator = mticker.FixedLocator( np.arange(-180,180+5,step=5) )
            gl.ylocator = mticker.FixedLocator( np.arange(-90,90+5,step=5) )
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            gl.xlabel_style = {'size': fontsize, 'color': 'gray'}
            gl.xlabel_style = {'size': fontsize, 'color': 'gray'}
            
        # CONSTRUCT: discrete colormap
    
        cmap = plt.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]        
        # cmaplist[0] = (.5, .5, .5, 1.0) # forces 1st entry to be grey
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)        
        bounds = np.linspace(0, 20, 21) # define the bins 
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N) # normalize

        # SCATTERPLOT                         
         
        im = plt.scatter(x=dm.lons, y=dm.lats, c=dm.counts, marker='s', 
                          transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, label='N(no normals)='+str(len(dm.counts)) ) 
        cb = plt.colorbar(im, orientation="vertical", shrink=0.5, extend='max')
        cb.ax.tick_params(labelsize=fontsize)
        cb.set_label(label = r'Number of stations without normals', size=fontsize)        
        plt.title(titlestr, fontsize=fontsize, pad=10)
        plt.savefig(figstr, dpi=300, bbox_inches='tight')
        plt.close('all')
        
    if plot_nonormals_stations_world == True:
        
        #------------------------------------------------------------------------------
        # PLOT: nonormals stations on world map
        #------------------------------------------------------------------------------
    
        print('plot_nonormals_station_locations ...')
            
        figstr = 'short-segment-stations-' + str(t_start) + '-' + str(t_end) + '.png'
        titlestr = 'GloSAT.p04: stations: ' + str(t_start) + '-' + str(t_end)
        
        fig  = plt.figure(figsize=(15,10))
        if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
        if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
        if projection == 'robinson': p = ccrs.Robinson(central_longitude=0)
        if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
        if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
        if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
        if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
        if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(central_longitude=-30); threshold = 0
        if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
        if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0
        if projection == 'winkeltripel': p = ccrs.WinkelTripel(central_longitude=0); threshold = 0
        ax = plt.axes(projection=p)    
        ax.set_global()
        ax.stock_img()
        if show_gridlines == True:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='grey', alpha=0.5, linestyle='-')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.left_labels = False
            gl.right_labels = False
            gl.xlines = True
            gl.ylines = True
            gl.xlocator = mticker.FixedLocator( np.arange(-180,180+5,step=5) )
            gl.ylocator = mticker.FixedLocator( np.arange(-90,90+5,step=5) )
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            gl.xlabel_style = {'size': fontsize, 'color': 'gray'}
            gl.xlabel_style = {'size': fontsize, 'color': 'gray'}
                                         
        plt.scatter(x=ds_lon, y=ds_lat, color="red", s=2, marker='o', alpha=1,
                    transform=ccrs.PlateCarree(), label='N(no normals)='+str(len(ds_lon)) ) 
#        plt.scatter(x=da_lon, y=da_lat, color="blue", s=2, marker='o', alpha=1,
#                    transform=ccrs.PlateCarree(), label='N(normals)='+str(len(da_lon)) ) 
        plt.legend(loc='lower left', bbox_to_anchor=(0, -0.15), markerscale=6, facecolor='lightgrey', framealpha=1, fontsize=14)    
        plt.title(titlestr, fontsize=fontsize, pad=10)
        plt.savefig(figstr, dpi=300, bbox_inches='tight')
        plt.close('all')

#------------------------------------------------------------------------------
# PLOT: number of short-segment series per time window
#------------------------------------------------------------------------------

sns.reset_orig()

figstr = 'short-segment-stations-evolution.png'
titlestr = 'GloSAT.p04: decadal coverage of stations with and without normals'
                        
fig, ax = plt.subplots(figsize=(15,10))     
plt.plot(edges[0:-1], n_all, marker='.', ls='-', lw=2, color='black', label='all stations')
plt.fill_between(edges[0:-1], n_all, color='black', alpha=0.2)
plt.plot(edges[0:-1], n_normals, marker='.', ls='-', lw=2, color='blue', label='with normals')
plt.plot(edges[0:-1], n_nonormals, marker='.', ls='-', lw=2, color='red', label='no normals')
plt.fill_between(edges[0:-1], n_normals, n_all, color='black', alpha=0.1)
plt.fill_between(edges[0:-1], n_nonormals, n_normals, color='blue', alpha=0.2)
plt.fill_between(edges[0:-1], n_nonormals, color='red', alpha=0.2)
ylimits = ax.get_ylim()
ax.axvline(x=1850, ls='dotted', lw=2, color='black')
ax.text(1851, ylimits[1]*0.95, 'Start of CRUTEM5.0.1', fontsize=fontsize)
#plt.yscale('log')
ax1 = plt.gca()
ax2 = ax.twinx()
ax2.plot(edges[0:-1], n_nonormals_proportion, marker='.', ls='-', lw=2, color='green')
ax2.fill_between(edges[0:-1], n_nonormals_proportion, color='green', alpha=0.2, hatch='///', zorder=2, fc='c')
ax1.set_xlabel('Year', fontsize=fontsize)
ax1.set_ylabel('Number of stations', fontsize=fontsize)
ax1.legend(loc='upper left', fontsize=fontsize)    
ax1.xaxis.grid(False, which='major')      
ax1.tick_params(labelsize=fontsize)    
ax2.set_ylabel('no normals proportion', fontsize=fontsize, color='green')
ax2.tick_params(labelsize=fontsize, colors='green')    
ax2.spines['right'].set_color('green')
plt.title(titlestr, fontsize=fontsize, pad=10)
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------
print('** END')

