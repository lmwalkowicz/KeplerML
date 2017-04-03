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


identifier = raw_input('Enter the identifier of the data: ')

# nclusters could (someday *should*) be obtained through the optimalK.py script 
nclusters = int(raw_input('Enter the number of clusters expected: '))

# byFeature is an array organized by feature. [[all feature 1 data],[all feature 2 data],...] 
# Useful for plotting and finding extrema.
def byFeature(X):
    return [[X[i][j] for i in range(len(X))] for j in range(len(X[0]))]

def bounding_box(X):
    # X is the data that comes in, it's organized by lightcurve[[all features for lc 1],[features for lc2],...]
    # To find minima we need to consider all points for each feature seperate from the other features.
    Xbyfeature = byFeature(X)

    # xmin/xmax will be an array of the minimum/maximum values of the features
    xmin=[]
    xmax=[]
    # Create the minimum vertex, and the maximum vertex for the box
    for feature in range(60):
        xmin.append(min(Xbyfeature[feature]))
        xmax.append(max(Xbyfeature[feature]))
        
    return (xmin,xmax)

def KMeans_clusters(data,nclusters):
    # Run KMeans, get clusters
    npdata = np.array(data)
    est = KMeans(n_clusters=nclusters)
    est.fit(npdata)
    clusters = est.labels_
    centers = est.cluster_centers_
    return clusters, centers

def seperatedClusters(data,nclusters,clusters):
    """
    Args:
    data - all data, organized by lightcurve 
    nclusters - number of clusters
    clusters - cluster labels
    
    Purpose: Create arrays dataByCluster and clusterIndexes containing data seperated by
    by cluster.
    """
    dataByCluster = []
    clusterIndexes = []
    
    # Will try to stick to the following:
    # cluster i, lightcurve j, feature k
    for i in range(nclusters):
        # Keeping track of which points get pulled into each cluster:
        clusterIndexes.append([j for j in range(len(data)) if clusters[j]==i])
        # Separating the clusters out into their own arrays (w/in the cluster array)
        dataByCluster.append([data[clusterIndexes[i][j]] for j in range(len(clusterIndexes[i]))])
        
        #Alternatively: (might have had some issues with the above, not sure...)
        #dataByCluster.append([])
        #for j in range(len(clusterIndexes[i])):
        #    dataByCluster[i].append(data[clusterIndexes[i][j]])
    return dataByCluster, clusterIndexes

def distances(pointsForDistance,referencePoint):
    """
    Args:
    cluster - array of any number of points, each point an array of it's features
    centerloc - array of the feautres for a single point
    
    Purpose: 
    This will calculate the distances of a group of points to a given point.
    """
    distFromCenter = [0 for j in range(len(pointsForDistance))] # reinitializing for each cluster
    # loop for each lightcurve of the cluster
    for j in range(len(pointsForDistance)):
        dataloc=pointsForDistance[j] # coordinates of the datapoint
        sqrd=0
        # loop for each feature of the lightcurve 
        for k in range(len(referencePoint)):
            sqrd+=pow(dataloc[k]-centerloc[k],2) # (x-x0)^2+(y-y0)^2+...
        distance = pow(sqrd,0.5) # sqrt((x-x0)^2+(y-y0)^2+...)
        distFromCenter[j]=distance 
    return distFromCenter

def beyondCutoff(cutoff,distances):
    """
    returns: outliers and typical arrays
        outlier array contains indexes of the specific cluster array that are beyond the cutoff
        typical array contains single index of the cluster array that is nearest the center
    
    Args:
    cutoff - a number, everything beyond this number is considered an outlier
    distances - array containing distances to each lightcurve point from the center for a given cluster
    """
    outliers=[j for j in range(len(distances)) if distances[j]>=cutoff] # recalculated for each cluster
    typical=[j for j in range(len(distances)) if distances[j]==min(distances)]
    return outliers,typical

def outliers(data,clusters,centers,files):
    """
    Args:
    data - all the data
    clusters - the cluster labels from kmeans. DBSCAN will likely require different methodology
    centers - locations of the cluster centers
    
    Purpose:
    Separate out the data on the edge of the clusters which are the most likely anomalous data.
    
    """
    
    nclusters = len(centers)

    # Sanity check, if this isn't an outlier than something is wrong.
    """controlPoint = np.array([10000 for i in range(len(data[0]))])
    
    data=np.append(data,[controlPoint],axis=0)
    clusters=np.append(clusters,[1],axis=0)"""
    
    """
    Initializing arrays
    """
    
    cluster, clusterIndexes = seperatedClusters(data,nclusters,clusters)
    twoSigma = [] #probably doesn't need an array 
    distFromCenter = []
    allTypical=[]
    allOutliers=[]
    # Will try to stick to the following:
    # cluster i, lightcurve j, feature k
    for i in range(nclusters):
                
        """
        ========== Checking density of clusters ============
        Given that some of the clusters may have 0 spread in a given feature, I'm not confident that this is meaningful.
        """
        (xmin,xmax)=bounding_box(cluster[i]) # get the minimum and maximum values for both
        diff=[xmax[n]-xmin[n] for n in range(len(xmin))]
        volOfClusterBB=1.0
        for n in diff:
            if n!=0:
                volOfClusterBB*=n
        densOfCluster=len(cluster[i])/volOfClusterBB
        """print("Cluster: %s, Density: %s" %(i,densOfCluster))
        print volOfClusterBB"""
        
        """
        ========== Finding points outside of the cutoff ===========
        """
        # not currently using this cutoff.
        twoSigma.append(2*np.std(cluster[i]))
        
        """
            ==== Calculating distances to each point ====
        """
        # Calculate distances to each point in each cluster to the center of its cluster
        distFromCenter.append(distances(cluster[i],centers[i]))
        
        """
            ==== Finding outliers and the standard (defined by the closest to the center) ====
        """
        cutoff=.5*max(distFromCenter[i])
        
        outliers,typical = beyondCutoff(cutoff,distFromCenter[i])
        
        # Arbitrarily large integer to indicate a new cluster is being considered.
        # will need made bigger if considering 1 billion+ lightcurves
        allTypical.append(987654321)
        # Add typical lightcurve to list. Only 1 produced at present, but this may change. The following
        # accounts for adding more typicals later.
        for j in typical:    
            allTypical.append(clusterIndexes[i][j])
            
        # place outliers from this cluster into general outlier list        
        allOutliers.append(987654321)
        if len(outliers)==0:
            print("none")
        else:
            for j in outliers:
                allOutliers.append(clusterIndexes[i][j])
    
    ncluster=1            
    print("Typical")
    for i in allTypical:
        if i == 987654321:
            print('')
            print("Cluster: %s"%cluster)
            cluster+=1
        else:
            print files[i]
            outputfile = open('c%soutlierfilelist'%(cluster-1),'w')
            outputfile.write('%s\n'%files[i])
    print("Outliers")
    ncluster=1
    for i in allOutliers:
        if i == 987654321:
            print('')
            print("Cluster: %s"%cluster)
            cluster+=1
        else:
            print files[i]
            with open('c%soutlierfilelist'%(cluster-1),'a') as outputfile:
                outputfile.write('%s\n'%files[i])
        
    return allOutliers,allTypical

    
"""ffeaturesID = str(identifier)+'dataByFeature.npy'
ffeatures = np.load(ffeaturesID)"""
dataID = str(identifier)+'dataByLightCurve.npy'
data = np.load(dataID)
ffeatures=[[data[i][j] for i in range(len(data))] for j in range(len(data[0]))]
filelist = str(identifier)+'filelist'
files = [line.strip() for line in open(filelist)]

clusterlabels,outliers,typical=KMeans_clusters(data,nclusters)

dataByCluster,clusterIndexes = seperatedClusters(data,nclusters,clusters)

for i in range(nclusters):
    for j in clusterIndexes[i]:
        outputfile = open('c%soutlierfilelist'%(ncluster+1),'w')
        outputfile.write('%s\n'%files[j])

for i in outliers:
    if i == 987654321:
        print('')
        print("Cluster: %s"%cluster)
        cluster+=1
    else:
        print files[i]
        with open('c%soutlierfilelist'%(cluster-1),'a') as outputfile:
            outputfile.write('%s\n'%files[i])