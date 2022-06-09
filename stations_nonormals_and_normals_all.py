#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: stations_nonormals_and_normals_all.py
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
import pandas as pd
import pickle
from datetime import datetime
import nc_time_axis
import cftime

# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
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

plot_station_locations = True
show_gridlines = True
projection = 'mollweide'        

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
    
dt_lon = dt.groupby('stationcode').mean()['stationlon']
dt_lat = dt.groupby('stationcode').mean()['stationlat']
da_lon = da.groupby('stationcode').mean()['stationlon']
da_lat = da.groupby('stationcode').mean()['stationlat']    
                
#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

if plot_station_locations == True:
    
    #------------------------------------------------------------------------------
    # PLOT: stations on world map
    #------------------------------------------------------------------------------

    print('plot_station_locations ...')
        
    figstr = 'stations_nonormals_and_normals_all.png'
    titlestr = 'GloSAT.p04: N(stations)=' + str(len(dt['stationcode'].unique()))
     
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
        ax.gridlines()     
        if projection == 'platecarree':
            ax.set_extent([-180, 180, -90, 90], crs=p)    
            gl = ax.gridlines(crs=p, draw_labels=False, linewidth=1, color='k', alpha=1.0, linestyle='-')
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlines = True
            gl.ylines = True
            gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
            gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': fontsize}
            gl.ylabel_style = {'size': fontsize}                                   
    plt.scatter(x=dt_lon, y=dt_lat, color="red", s=2, marker='o', alpha=1,
                transform=ccrs.PlateCarree(), label='Global: N(no normals)='+str(len(dt['stationcode'].unique()) - len(da['stationcode'].unique()) )) 
    plt.scatter(x=da_lon, y=da_lat, color="blue", s=2, marker='o', alpha=1,
                transform=ccrs.PlateCarree(), label='Global: N(normals)='+str(len(da['stationcode'].unique())) ) 
    plt.legend(loc='lower left', bbox_to_anchor=(0, -0.15), markerscale=6, facecolor='lightgrey', framealpha=1, fontsize=14)    
    plt.title(titlestr, fontsize=fontsize, pad=10)
    plt.savefig(figstr, dpi=300, bbox_inches='tight')
    plt.close('all')

#------------------------------------------------------------------------------
print('** END')

