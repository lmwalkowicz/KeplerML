import ast
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
from sklearn.cluster import DBSCAN
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
import sys
import cPickle as pickle
if sys.version_info[0] < 3:
    from Tkinter import Tk
else:
    from tkinter import Tk

from tkFileDialog import askopenfilename,askdirectory

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])

def bounding_box(X):
    # X is the data that comes in, it's organized by lightcurve[[all features for lc 1],[features for lc2],...]
    
    # Xbyfeature is an array organized by feature. [[all feature 1 data],[all feature 2 data],...] 
    Xbyfeature = [[X[i][j] for i in range(len(X))] for j in range(len(X[0]))]

    # xmin/xmax will be an array of the minimum/maximum values of the features
    xmin=[]
    xmax=[]
    for feature in range(60):
        xmin.append(min(Xbyfeature[feature]))
        xmax.append(max(Xbyfeature[feature]))
        
    return (xmin,xmax)

def montecarlogauss(n):
    xmin = np.load('xmin.npy')
    xmax = np.load('xmax.npy')
    Xbn = [random.uniform(xmin[j],xmax[j]) for j in range(60)]
    return Xbn
        
def gap_statistic(k):
    X = np.load('tempdata.npy')
    (xmin,xmax) = bounding_box(X)
    
    # temporary npy files for multiprocessing
    np.save('xmin',xmin)
    np.save('xmax',xmax)
    
    mu, clusters = find_centers(X,k)
    Wks = np.log(Wk(mu, clusters))
    # Create B reference datasets
    B = 10
    BWkbs = np.zeros(B)
    for i in range(B):
        # create random distribution of n points in the bounding hypercube, this is time consuming
        # with hundreds of thousands of points. Multiprocessing solution? see comments in section, might work
        Xb = []
            
        numcpus = cpu_count()
        if numcpus > 1:
            # Leave 1 cpu open for system tasks.
            usecpus = numcpus-1
        else:
            usecpus = 1
            
        p = Pool(usecpus)
        Xb = p.map(monteCarloGauss,range(len(X)))
        p.close()
        p.terminate()
        p.join()
        """
        for n in range(len(X)):
            Xb.append([random.uniform(xmin[j],xmax[j]) for j in range(60)])
        """
        Xb = np.array(Xb)
        mu, clusters = find_centers(Xb,k)
        BWkbs[i] = np.log(Wk(mu, clusters))
    
    os.remove('xmin.npy')
    os.remove('xmax.npy')
    
    Wkbs = sum(BWkbs)/B
    sk = np.sqrt(sum((BWkbs-Wkbs)**2)/B)*np.sqrt(1+1/B)
    gs = Wkbs - Wks
    
    return gs,sk

def optimalK(X):
    # resaving the data as tempdata (generic name) so that it can be read by the main loop
    np.save('tempdata',X)

    # Dispersion for real distribution
    ### Adjust range of clusters tried here:
    ks = range(1,10)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    gs = np.zeros(len(ks))
    for k in ks:
        gapstat = gap_statistic(k)
    for indk,k in enumerate(ks):
        gs[indk] = gapstat[indk][0]
        sk[indk] = gapstat[indk][1]   
    """
    if __name__ == '__main__':    
        # probably not the best way to make this run faster... but it makes it 10x faster for now
        p = Pool(10)
        gapstat = p.map(gap_statistic,ks)
        for indk,k in enumerate(ks):
            gs[indk] = gapstat[indk][0]
            sk[indk] = gapstat[indk][1]
        p.close()
        p.terminate()
        p.join()
    """        
    # deleting the temporary data file.
    os.remove('tempdata.npy')
    return min([k for k in range(1,len(ks)-1) if gs[k]-(gs[k+1]-sk[k+1]) >= 0])

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
output = askopenfilename()
print(output)
if output:
    outputfile = open(output,'r+') # show an "Open" dialog box and return the path to the selected file
    outputdata = []
    print("Importing Data...")
    while True:
        try:
            o = pickle.load(outputfile)
        except EOFError:
            break
        else:
            outputdata.append(o)
    outputfile.close()
    
    if len(outputdata) == 1:
        outputdata = outputdata[0]
        
    files = [outputdata[i][0] for i in range(len(outputdata))\
             if not np.isnan(outputdata[i][1:]).any() and not np.isinf(outputdata[i][1:]).any()]
    data = np.array([outputdata[i][1:] for i in range(len(outputdata))\
             if not np.isnan(outputdata[i][1:]).any() and not np.isinf(outputdata[i][1:]).any()])

    print("Finding optimal k...")
    print optimalK(data)