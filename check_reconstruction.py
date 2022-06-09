#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: check_reconstruction.py
#------------------------------------------------------------------------------
# Version 0.1
# 9 June, 2022
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

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

nclusters = 40

#------------------------------------------------------------------------------
# LOAD: input dataframe and reconstructed dataframe
#------------------------------------------------------------------------------

df_temp_reconstructed = pd.read_pickle('OUT/' + str(nclusters).zfill(2) + '/df_temp_reconstructed.pkl', compression='bz2')
df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')

#------------------------------------------------------------------------------
# ARCHIVE: equality tests
#------------------------------------------------------------------------------

A = df_temp_reconstructed.copy()
B = df_temp.copy()
C = np.subtract( A.iloc[:,1:13], B.iloc[:,1:13])

test_equal_shape = np.subtract( A.shape, B.shape ) 
test_equal_values = np.nansum( C, axis=0 )

s1 = A.stationcode.unique()
s2 = B.stationcode.unique()
idx_AB = list( set( s1 ) - set( s2 ) )  # stationcodes in s1 not in s2
idx_BA = list( set( s2 ) - set( s1 ) )  # stationcodes in s2 not in s1

#test_equal_shape = np.array_equal(A,B) # test if same shape, same elements values 				--> fails due to mixed numeric & str types
#test_equiv = np.array_equiv(A,B)  		# test if broadcastable shape, same elements values 	--> fails due to mixed numeric & str types
#test_close = np.allclose(A,B,...) 		# test if same shape, elements have close enough values	--> fails due to mixed numeric & str types

print('TEST: shape difference:', test_equal_shape)
print('TEST: columnar total difference:', test_equal_values)
print('TEST: stationcodes in A not in B:', idx_AB)
print('TEST: stationcodes in B not in A:', idx_BA)

#TEST: shape difference: [0 0]
#TEST: columnar total difference: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#TEST: stationcodes in A not in B: []
#TEST: stationcodes in B not in A: []
 
#------------------------------------------------------------------------------
print('** END')

