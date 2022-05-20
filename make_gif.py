#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Quick animated GIF maker
"""

#------------------------------------------------------------------------------
# PROGRAM: make_gif.py
#------------------------------------------------------------------------------
# Version 0.1
# 28 February, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

import os, glob
import imageio

#----------------------------------------------------------------------------
# MAKE GIF
#----------------------------------------------------------------------------

use_reverse_order = False

nclusters = 50
png_dir = 'OUT/' + str(nclusters).zfill(2) + '/cluster-map/*.png'
gif_str = str(nclusters).zfill(2) + '-clusters.gif'
mp4_str = str(nclusters).zfill(2) + '-clusters.mp4'

if use_reverse_order == True:
    a = glob.glob(png_dir)
    images = sorted(a, reverse=True)
else:
    images = sorted(glob.glob(png_dir))

var = [imageio.imread(file) for file in images]
imageio.mimsave(gif_str, var, fps = 1)

#----------------------------------------------------------------------------
# CLI --> MAKE GIF & MP4
#----------------------------------------------------------------------------

# PNG --> GIF:
# convert -delay 10 -loop 0 png_dir gif_str

# GIF --> MP4
# ffmpeg -i gif_str -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" mp4_str


# -----------------------------------------------------------------------------
print('** END')
