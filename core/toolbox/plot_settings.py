#!/usr/bin/python

# Module: plot_settings.py
# Author: Varun Hiremath <vh63@cornell.edu>
# Created: Thu,  2 Apr 2009 05:06:31 -0400

import math
import numpy

from my_axes import MyAxes, MyAxes3D
from toolbox import pylab

import pprint
pp=pprint.pprint

# Symbols
symbols = ['-','--','-.',':','.',',','o','^','v','<','>','s','+','x','D','d','1','2','3','4','h','H','p']
# Symbols + line
lps = [k+'-' for k in [',','.','o','^','v','<','>','s','+','x','D','d','1','2','3','4','h','H','p']]
# Colors
colors= ['b','g','r','c','m','y','k','w']

fontsize=range(1,31)
fontHightPixels=numpy.array([3, 4, 6, 7, 9, 11, 12, 13, 16, 17, 19, 20, 22, 
                             24, 25, 26, 28, 30, 32, 33, 34, 37, 38, 39, 41, 
                             43, 45, 46, 47, 49])


def measure_font():
    import Tkinter as tk
    import tkFont
    root = tk.Tk()
    pixels=[]
    for i in range(1,31): 
        font = tkFont.Font(root=root, family='Times',size=i)
        pixels.append(font.metrics('linespace'))
            

def get_figsize(w, h = None, **kwargs):
    if h == None:
        inches_per_pt = 1.0/72.0                # Convert pt to inch
        golden_mean = (math.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_width = w*inches_per_pt  # width in inches
        fig_height = w*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]      # exact figsize
    else:
        inches_per_pt = 1.0/72.0                # Convert pt to inch
        fig_width  = w*inches_per_pt            # width in inches
        fig_height = h*inches_per_pt            # height in inches
        fig_size =  [fig_width,fig_height]      # exact figsize
    return fig_size

# Publishable quality image settings for 2-column papers
params0 = {'backend': 'eps',
          'axes.labelsize': 6,
          'text.fontsize': 6,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'legend.pad': 0.1,    # empty space around the legend box
          'legend.fontsize': 5,
          'lines.markersize': 3,
          'font.size': 6,
          #'text.usetex': True,
          'figure.figsize': get_figsize(250),
          'figure.frameon':False}

# Medium sized images
params1 = {'backend': 'eps',
          'axes.labelsize': 8,
          'text.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.pad': 0.1,     # empty space around the legend box
          'legend.fontsize': 8,
          'lines.markersize': 3,
          'font.size': 8,
          #'text.usetex': True,
          'figure.figsize': get_figsize(500),
          'figure.frameon':False}

# Large images (default)
params2 = {'backend': 'eps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'legend.pad': 0.2,     # empty space around the legend box
          'legend.fontsize': 10,
           'lines.markersize': 3,
          'font.size': 10,
          #'text.usetex': True,
          'figure.figsize': get_figsize(800),
          'axes.frameon':False}

# My images (default)
params3 = {'backend': 'eps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'legend.pad': 0.2,     # empty space around the legend box
          'legend.fontsize': 10,
          'lines.markersize': 3,
          'font.size': 10,
          #'text.usetex': True,
          'figure.figsize': get_figsize(800),
          'axes.frameon':False}

params4 = {'backend': 'eps',
          'axes.labelsize': 7,
          'text.fontsize': 7,
          'xtick.labelsize': 7,
          'ytick.labelsize': 7,
          'legend.pad': 0.1,     # empty space around the legend box
          'legend.fontsize': 7,
          'lines.markersize': 3,
          'font.size': 7,
          #'text.usetex': True,
          'figure.figsize': get_figsize(500),
          'figure.frameon':False}

params5 = {'backend': 'eps',
          'axes.labelsize': 12,
          'text.fontsize': 12,
          'title.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.pad': 0.2,    # empty space around the legend box
          'legend.fontsize': 12,
          'lines.markersize': 3,
          'font.size': 12,
          #'text.usetex': True,
          'figure.figsize': get_figsize(250),
          'figure.frameon':False}

params6 = {'backend': 'eps',
          'axes.labelsize': 20,
          'text.fontsize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'legend.pad': 0.2,     # empty space around the legend box
          'legend.fontsize': 20,
          'lines.markersize': 3,
          'font.size': 20,
          #'text.usetex': True,
          'figure.figsize': get_figsize(800),
          'axes.frameon':False}

params7 = {'backend': 'eps',
          'axes.labelsize': 20,
          'text.fontsize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'legend.pad': 0.2,     # empty space around the legend box
          'legend.fontsize': 20,
          'lines.markersize': 3,
          'font.size': 20,
          #'text.usetex': True,
          'figure.figsize': get_figsize(800),
          'axes.frameon':False}

def set_mode(pylab, mode='large', w = None, h = None, fontsize=10, **kwargs):
    pl=pylab
    if mode == "dynamic":
        ratio=33/400.*h
        fh=fontHightPixels[ratio>=fontHightPixels]
        fs=fh.size
        params7 = {'backend': 'eps',
          'axes.labelsize': fs,
          'text.fontsize': kwargs.get('text.fontsize',fs),
          'xtick.labelsize': fs,
          'ytick.labelsize': fs,
          'legend.pad': 0.2,     # empty space around the legend box
          'legend.fontsize': fs,
          'lines.markersize': 3,
          'font.size': fs,
          #'text.usetex': True,
          'figure.figsize': get_figsize( kwargs.get('dpi',800), w = w, h = h),
          'axes.frameon':False}
    #matplotlib.use("WXAgg") # do this before pylab so you don'tget the default back end.

    if mode == "by_fontsize":
        fs=fontsize
        params8 = {'backend': 'Agg',
                   'axes.titlesize':kwargs.get('title_fontsize',fs),
          'axes.labelsize': fs,
          'text.fontsize': kwargs.get('text_fontsize',fs),
          'text.usetex' : kwargs.get('text_usetex',False),
          'xtick.labelsize': fs,
          'ytick.labelsize': fs,
          'legend.pad': 0.2,     # empty space around the legend box
          'legend.fontsize': kwargs.get('legend_fontsize',fs),
          'lines.markersize': 3,
          'lines.linewidth':kwargs.get( 'linewidth', 2.),
          'font.size': kwargs.get('font_size',fs),
          'mathtext.default':'sf', # Use sans-serif font family for math text
                                   # priority list, check mathtext for the other
                                   # font alternatives. Defailt is italic serif.
                                   # Here sf is serif-sans
          #'text.usetex': True,
          'figure.figsize': get_figsize( w, h),
          'figure.dpi':  kwargs.get('dpi',72),       # dots per inches, increase and font size follows
          'savefig.dpi': kwargs.get('dpi',72),
          'axes.frameon':False,
          'path.simplify':False}    
        pl.rcParams.update(params8)
        
    if mode == 'my_large':
        params6['figure.figsize'] = get_figsize( 800, w = w, h = h)
        pl.rcParams.update(params6)
    if mode == 'my_publish':
        params3['figure.figsize'] = get_figsize( 800, w = w, h = h)
        pl.rcParams.update(params3)
    elif mode == 'my_medium':
        params5['figure.figsize'] = get_figsize( 500, w = w, h = h)
        pl.rcParams.update(params5)
    elif mode == 'my_low':
        params4['figure.figsize'] = get_figsize( 250, w = w, h = h)
        pl.rcParams.update(params4)    
    elif mode == "publish":
        pl.rcParams.update(params0)
    elif mode == "medium":
        pl.rcParams.update(params1)
    #else:
    # pl.rcParams.update(params2)

def set_figsize(fig_width_pt):
    pylab.rcParams['figure.figsize'] = get_figsize(fig_width_pt)


def get_figure( n_rows, n_cols, w, h=None, fontsize=12, order='col', projection=None, **kwargs):
#     import os
#     if not os.environ.get('DISPLAY'):
#         pylab.ioff()
    
    set_mode(pylab, 'by_fontsize', w, h, fontsize, **kwargs)
#     pylab.rcParams.update({'backend':'Agg'})
#     pp(pylab.rcParams)
#     import matplotlib
#     matplotlib.use('Agg')
    fig = pylab.figure( facecolor = 'w' )
    
    hight=1./n_rows*kwargs.get('frame_hight_y',0.7)
    width=1./n_cols*kwargs.get('frame_hight_x',0.7)
    inbetw_row=(1-hight*n_rows)/(1+n_rows)
    inbetw_col=(1-width*n_cols)/(1+n_cols)
    step_row=hight+inbetw_row
    step_col=width+inbetw_col
    ypoints=list(numpy.linspace(inbetw_row, 1-step_row, n_rows))
    xpoints=list(numpy.linspace(inbetw_col, 1-step_col ,n_cols))
    ax_list=[]
    

    if projection=='3d':
        axes_class=MyAxes3D
    else:
        axes_class=MyAxes
    
    if order=='col':
        for y in reversed(ypoints):
            for x in xpoints:
                ax_list.append( axes_class(fig, [ x,  y,  width, hight ] ) ) 
    
    if order=='row':
        for x in xpoints:
            for y in reversed(ypoints):
                ax_list.append( axes_class(fig, [ x,  y,  width, hight ] ) )                 
    return fig, ax_list


def shift(flag, axs, shift, n_rows=1, n_cols=1):
    assert n_rows*n_cols==len(axs)
    shift_vert=shift/n_rows
    shift_hor=shift/n_cols
    k=0
    for i in range(n_rows):
        for j in range(n_cols):
            ax=axs[k]
            k+=1 
            
            if flag=='upp':
                box = ax.get_position()
                ax.set_position([box.x0, box.y0+shift_vert*i, box.width, box.height*(1-shift_vert)])

            if flag=='down':
                box = ax.get_position()
                ax.set_position([box.x0, box.y0-shift_vert*(n_rows-i), 
                                 box.width, box.height*(1-shift_vert)])

            if flag=='left':
                box = ax.get_position()
                ax.set_position([box.x0+shift_hor*j, box.y0, box.width*(1-shift_hor), box.height])
            if flag=='right':
                box = ax.get_position()
                ax.set_position([box.x0+shift_hor*(n_cols-j), box.y0, box.width*(1-shift_hor), box.height])

import unittest

class TestPlotSettings(unittest.TestCase):     
    def setUp(self):
        self.fname='/home/mikael/results/papers/inhibition/network/unittest/plot_settings/'
    def test_get_figure(self):
        fig1, axs=get_figure( 2, 2, 100, 100, 12, order='col', projection=None, **{'dpi':72})
        fig1.savefig(self.fname+'fig_1')
        fig2, axs=get_figure( 2, 2, 1200, 1200, 12, order='col', projection=None, **{'dpi':72})

        fig2.savefig(self.fname+'fig_2')
# show_phase_diff(d, models=models_coher)

if __name__ == '__main__':
    test_classes_to_run=[
                         TestPlotSettings
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)

