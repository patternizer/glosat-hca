#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: iter-runtime-curve.py
#------------------------------------------------------------------------------
# Version 0.1
# 6 December, 2022
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

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 16

glosat_version = 'GloSAT.p04c.EBC'

#------------------------------------------------------------------------------
# LOAD: JASMIN runtime array
#------------------------------------------------------------------------------

df = pd.read_csv('DATA/time-iter-n-t.txt')

#------------------------------------------------------------------------------
# PLOT: runtime as a function of number of stations in USA
#------------------------------------------------------------------------------

figstr = 'lek-iter-runtime-curve-ncycles10-logarithmic.png'
titlestr = glosat_version + ': JASMIN runtime as a function of number of stations in USA clusters (logarithmic)'
                        
fig, ax = plt.subplots(figsize=(15,10))     
plt.plot( df.n, np.log10(df.t/60), marker='o', ls='-', lw=0.5)
plt.tick_params(labelsize=fontsize)    
plt.xlabel('Number of stations in cluster: ' + r'$log_{2}$(n)', fontsize=fontsize)
plt.ylabel('JASMIN runtime: ' + r'$log_{10}$(minutes)', fontsize=fontsize)
plt.tick_params(labelsize=fontsize, colors='green')    
plt.title(titlestr, fontsize=fontsize, pad=10)
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close('all')

figstr = 'lek-iter-runtime-curve-ncycles10.png'
titlestr = glosat_version + ': JASMIN runtime as a function of number of stations in USA clusters'
                        
fig, ax = plt.subplots(figsize=(15,10))     
plt.plot( 2**df.n, df.t/60, marker='o', ls='-', lw=0.5)
plt.tick_params(labelsize=fontsize)    
plt.xlabel('Number of stations in cluster', fontsize=fontsize)
plt.ylabel('JASMIN runtime [minutes]', fontsize=fontsize)
plt.tick_params(labelsize=fontsize, colors='green')    
plt.title(titlestr, fontsize=fontsize, pad=10)
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------
print('** END')

