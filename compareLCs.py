# L.M. Walkowicz
# Rewrite of Revant's feature calculations, plus additional functions for vetting outliers

import random
import numpy as np
np.set_printoptions(threshold='nan')
import scipy as sp
from scipy import stats
import pyfits
import math
import pylab as pl
import matplotlib.pyplot as plt
import heapq
from operator import xor
import scipy.signal
from numpy import float64
#import astroML.time_series
#import astroML_addons.periodogram
#import cython
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN,Ward
from numpy.random import RandomState
#rng = RandomState(42)
import itertools
import commands
# import utils
import itertools
#from astropy.io import fits
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons

filelist=str(raw_input('Enter name of file list: '))
#identifier=str(raw_input('Enter a unique identifier: '))

def read_revant_pars(parfile):
    pars = [line.strip() for line in open(parfile)]
    pararr = np.zeros((len(pars), 60))
    for i in range(len(pars)):
        pararr[i] = np.fromstring(pars[i], dtype=float, sep=' ')
    return pararr

def read_kepler_curve(file):
    lc = pyfits.getdata(file)
    t = lc.field('TIME')
    f = lc.field('PDCSAP_FLUX')
    err = lc.field('PDCSAP_FLUX_ERR')
    nf = f / np.median(f)
 
    nf = nf[np.isfinite(t)]
    t = t[np.isfinite(t)]
    t = t[np.isfinite(nf)]
    nf = nf[np.isfinite(nf)]

    return t, nf, err

def plot_kepler_curve(t, nf):
    fig, ax = plt.subplots(figsize=(5, 3.75))
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel(r'${\rm Time (Days)}$', fontsize=20)
    ax.set_ylabel(r'${\rm \Delta F/F}$', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.plot(t, nf, 'o',markeredgecolor='none', color='blue', alpha=0.2)
    plt.plot(t, nf, '-',markeredgecolor='none', color='blue', alpha=1.0)
    plt.show()

def plot_curves(tArray,nfArray,files):
    
    # tArray should be an array with time data for each file [[file1 t0,t1,t2,...],[file2 t0,t1,t2,...],...]
    tdict= dict(zip(files,tArray))
    # nfArray contains flux data same format as the tArray
    nfdict= dict(zip(files,nfArray))
    # files should contain an array of the lightcurve names
    
    # labellist keeps track of what each axis should be labelled, initialized with first 3 features
    xlabel = files[0]
    
    # Plot initializing stuff
    fig,ax = plt.subplots(figsize=(16, 6))
    
    xaxis=tArray[0]
    yaxis=nfArray[0]
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(xaxis, yaxis, 'o',markeredgecolor='none', color='blue', alpha=0.2)
    plt.plot(xaxis, yaxis, '-',markeredgecolor='none', color='blue', alpha=1.0)

    plt.subplot(212)
    p1,=plt.plot(xaxis, yaxis, 'o',markeredgecolor='none', color='blue', alpha=0.2)
    p2,=plt.plot(xaxis, yaxis, '-',markeredgecolor='none', color='blue', alpha=1.0)

    plt.subplots_adjust(left=0.3)
    plt.xlabel(r'${\rm Time (Days)}$', fontsize=20)
    plt.title(files[0],fontsize=20)
    plt.ylabel(r'${\rm \Delta F/F}$', fontsize=20)

    """
    Matplotlib radiobuttons format poorly when there are more than ~5 buttons, this is an issue that will be resolved at some point in the future.
    For the time being, I'm making a bunch of radiobutton sections with 5 buttons apeice. Each column represents a differnt axis.
    The most recently clicked button will be the active one, so the color coding is less than useful.

    It's not pretty but it works.
    """

    rax1p = [[0.0,.92-i/12.0,0.1,1.0/12.0] for i in range(12)]
    
    rax10= plt.axes([0.0,0.45,0.2,0.3])
    
    #radio10 = RadioButtons(rax10, files[0:2])
    
    if len(files)<=5:
        radio10 = RadioButtons(rax10, files[0:len(files)])
    elif len(files)<=10:
        radio10 = RadioButtons(rax1p[0], files[0:5])
        radio11 = RadioButtons(rax1p[1], files[5:len(files)])
    elif len(files)<=15:
        radio10 = RadioButtons(rax1p[0], files[0:5])
        radio11 = RadioButtons(rax1p[1], files[5:10])
        radio12 = RadioButtons(rax1p[2], files[10:len(files)])
        
    def axis1(label):
        
        xaxis = tdict[label]
        yaxis = nfdict[label]
        
        p1.set_xdata(xaxis)
        p1.set_ydata(yaxis)
        p2.set_xdata(xaxis)
        p2.set_ydata(yaxis)
        """        plt.plot(xaxis, yaxis, 'o',markeredgecolor='none', color='blue', alpha=0.2)
        plt.plot(xaxis, yaxis, '-',markeredgecolor='none', color='blue', alpha=1.0)"""
        plt.subplot(212)
        plt.title(label,fontsize=20)
        plt.ylabel(r'${\rm \Delta F/F}$', fontsize=20)
        
        plt.draw()

    #radio10.on_clicked(axis1)
    if len(files)<=5:
        radio10.on_clicked(axis1)
    elif len(files)<=10:
        radio10.on_clicked(axis1)
        radio11.on_clicked(axis1)
    elif len(files)<=15:
        radio10.on_clicked(axis1)
        radio11.on_clicked(axis1)
        radio12.on_clicked(axis1)
    
    plt.show()
    
files = [line.strip() for line in open(filelist)]

tArr = []
nfArr = []
for f in files:
    t,nf,err=read_kepler_curve(f)
    tArr.append(t)
    nfArr.append(nf)

plot_curves(tArr,nfArr,files)

