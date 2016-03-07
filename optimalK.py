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

identifier = raw_input('Enter the identifier of the data: ')

dataID = str(identifier)+'dataByLightCurve.npy'
data = np.load(dataID)

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

def Xbb(X):
    (xmin,xmax) = bounding_box(X)
    Xb = [random.uniform(xmin[j],xmax[j]) for j in range(60)]
    return Xb

def gap_statistic(X):
    (xmin,xmax) = bounding_box(X)
    # Dispersion for real distribution
    ### Adjust range of clusters tried here:
    ks = range(1,10)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = np.zeros(B)
        for i in range(B):
            xfeedformulti=[X for i in range(len(X))]
            if __name__ == '__main__':    
                p = Pool(7)
                Xb = p.map(Xbb,xfeedformulti)
                p.close()
                p.terminate()
                p.join()
            #Xb = [[random.uniform(xmin[j],xmax[j]) for j in range(60)] for n in range(len(X))]
            #for n in range(len(X)):
            #    Xb.append([random.uniform(xmin[j],xmax[j]) for j in range(60)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)

def optimalK(data):
    ks, logWks, logWkbs, sk = gap_statistic(data)
    gs = logWkbs - logWks
    return min([k for k in range(1,len(ks)-1) if gs[k]-(gs[k+1]-sk[k+1]) >= 0])

print optimalK(data)