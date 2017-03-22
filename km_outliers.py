""" 
Creates clusters from processed data for any number of dimensions using k-means.

Requirement: Data in a pandas dataframe, files in a numpy array.
arranged as follows:
[str(filename),feature1,feature2,...,featureN]
"""

import random
import numpy as np
np.set_printoptions(threshold='nan')
from multiprocessing import Pool,cpu_count
import sys
from datetime import datetime
from numbapro import cuda
from sklearn.cluster import KMeans

@cuda.autojit
def cu_worker(x1, x2, mu, bmk):
    bx =cuda.blockIdx.x
    bw = cuda.blockDim.x
    tx = cuda.threadIdx.x
    j = tx+bx*bw
    
    if j>bmk.size:
        return
    num = 0
    for i in range(len(mu)):
        numold = num
        num = ((x1[j]-mu[i][0])**2+(x2[j]-mu[i][1])**2)**.5
        if num<numold or i==0:
            bmk[j]=i

def gpumulti(X,mu):
    device = cuda.get_current_device()
    
    n=len(X)
    X=np.array(X)
    x1 = np.array(X.T[0])
    x2 = np.array(X.T[1])
    
    bmk = np.arange(len(x1))
    
    mu = np.array(mu)
    
    dx1 = cuda.to_device(x1)
    dx2 = cuda.to_device(x2)
    dmu = cuda.to_device(mu)
    dbmk = cuda.to_device(bmk)
    
    # Set up enough threads for kernel
    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
        
    cu_worker[bpg,tpb](dx1,dx2,dmu,dbmk)
    
    bestmukey = dbmk.copy_to_host()
    
    return bestmukey

def cluster_points(X, mu):
    bestmukey = gpumulti(X,mu)
    
    bestmukey=np.array(bestmukey)
    X=np.array(X)
    clusters=[X[bestmukey==i] for i in range(min(bestmukey),max(bestmukey+1))]
    
    return clusters
 
def reevaluate_centers(mu, clusters):
    keys = enumerate(clusters)
    newmu=[np.mean(clusters[k[0]], axis = 0) for k in keys]
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):

    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    count = 0
    while not has_converged(mu, oldmu):
        # Sometimes it gets stuck and never converges (at least after 10+ hours...)
        # Try resetting mu and oldmu after a sufficient number of iterations, 
        # typically converges w/in 50 iterations 
        if count > 100:
            oldmu = random.sample(X,K)
            mu = random.sample(X,K)
            count = 0
        
        oldmu = mu
        # Assign all points in X to clusters
        
        clusters = cluster_points(X,mu)
        
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
            
    return(mu, clusters)

def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])

def bounding_box(X):
    # X is the data that comes in, it's organized by lightcurve[[all features for lc 1],[features for lc2],...]
    numFeats = len(X[0])
    numFiles = len(X)
    # Xbyfeature is an array organized by feature. [[all feature 1 data],[all feature 2 data],...] 
    Xbyfeature = [[X[i][j] for i in range(numFiles)] for j in range(numFeats)]

    # xmin/xmax will be an array of the minimum/maximum values of the features
    xmin=[]
    xmax=[]
    for ft in range(numFeats):
        xmin.append(min(Xbyfeature[ft]))
        xmax.append(max(Xbyfeature[ft]))
        
    return (xmin,xmax)
        
def gap_statistic(k):
    
    (xmin,xmax) = bounding_box(X)
        
    numFeats = len(xmin)
    mu, clusters = find_centers(X,k)

    Wks = np.log(Wk(mu, clusters))
    # Create B reference datasets
    B = 10
    BWkbs = np.zeros(B)

    for i in range(B):
        Xb = np.array([[random.uniform(xmin[j],xmax[j]) for j in range(numFeats)] for n in range(len(X))])
        mu, clusters = find_centers(Xb,k)    
        BWkbs[i] = np.log(Wk(mu, clusters))

    Wkbs = sum(BWkbs)/B
    sk = np.sqrt(sum((BWkbs-Wkbs)**2)/B)*np.sqrt(1+1/B)
    gs = Wkbs - Wks

    return gs,sk

def optimalK(X):

    # Dispersion for real distribution
    ### Adjust range of clusters tried here:
    ks = range(1,11)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    gs = np.zeros(len(ks))
    
    for indk,k in enumerate(ks):
        gs[indk],sk[indk] = gap_statistic(k)

    return min([k for k in range(1,len(ks)-1) if gs[k]-(gs[k+1]-sk[k+1]) >= 0])

"""
K-means Clustering
"""

def outliers(data,clusterLabels,centers):
    """
    Args:
    data - all the data
    clusters - the cluster labels from kmeans.
    centers - locations of the cluster centers
    
    Purpose:
    Separate out the data on the edge of the clusters 
    which are the most likely outliers.
    """
    # Will try to stick to the following:
    # cluster i, lightcurve j, feature k
    nclusters = len(centers)
    
    """
    Initializing arrays
    """
    
    dataByCluster = []
    clusterIndices = []
            
    allTypical=[]
    allOutliers=[]
    clusters = np.array(clusterLabels)
    for i in range(nclusters):

        """
            ==== Calculating distances to each point ====
        Calculate distances for each point to the center of its cluster
        """
        distFromCenter = [sum((data[pt]-centers[i])**2)**.5 for pt in clusterLabels[clusterLabels==i]]

        """
        ========== Finding points outside of the cutoff ===========
        """
        sigma = np.std(distFromCenter)
        cutoff = 2*sigma        
        """
            ==== Finding outliers and the standard (defined by the closest to the center) ====
        """
        
        # returns cluster indices of the outliers and the typical lcs
        outliers=[j[0] for j in enumerate(distFromCenter) if j[1]>=cutoff]
        typical = [j[0] for j in enumerate(distFromCenter) if j[1]==min(distFromCenter)]

        clusters[outliers]=-1
        clusters[typical]=0
        
    return clusters

"""
============ Start ============
"""

def kmeans_w_outliers(files,data,nclusters=1):
    # nclusters can be obtained through the optimalK.py script 
    # making a copy of the data dataframe
    X=np.array([np.array(data.loc[i]) for i in data.index])
    # Run KMeans, get clusters
    est = KMeans(n_clusters=nclusters)
    est.fit(X)
    clusters = est.labels_
    clusterLabels=clusters
    centers = est.cluster_centers_
    
    clusterLabels=outliers(X,clusterLabels,centers)
    clusterLabels = np.array(clusterLabels)
    numout = len(clusterLabels[clusterLabels==-1])
    if data.index.str.contains('8462852').any():
        tabbyInd = list(data.index).index('8462852')
        if clusterLabels[tabbyInd] == -1:
            print("Tabby has been found to be an outlier in k-means.")
        else:
            print("Tabby has NOT been found to be an outlier in k-means")

    print("There were %s outliers in %s clusters"%(numout,nclusters))
    
    return clusterLabels

if __name__=="__main__":
    """
    If this is run as a script, the following will parse the arguments it is fed, 
    or prompt the user for input.
    
    python keplerml.py path/to/filelist path/to/fits_file_directory path/to/output_file
    """
    # fl - filelist, a txt file with file names, 1 per line
    if sys.argv[1]:
        f = sys.argv[1]
    else:
        while not f:
            f = raw_input("Input path: ")
     
    print("Reading %s..."%f)
    
    df = pd.read_csv(f,index_col=0)
    
    if sys.argv[2]:
        of = sys.argv[2]
    else:
        of = raw_input("Output path: ")
    if not of:
        print("No output path specified, saving to output in local folder.")
        of = 'output'
    
    np.save(of,kmeans_w_outliers(df.index,df[['tsne_x','tsne_y']],1))
    print("Done.")