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

# nclusters could be obtained through the optimalK.py script
nclusters = int(raw_input('Enter the number of clusters expected: '))


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

def KMeans_clusters(data,nclusters):
    # Run KMeans, get clusters
    npdata = np.array(data)
    est = KMeans(n_clusters=nclusters)
    est.fit(npdata)
    clusters = est.labels_
    centers = est.cluster_centers_
    
    """controlPoint = np.array([10000 for i in range(len(data[0]))])
    
    data=np.append(data,[controlPoint],axis=0)
    clusters=np.append(clusters,[1],axis=0)"""
    """
    Initializing arrays
    """
    cluster = [[] for i in range(nclusters)]
    clusterIndexes = [[] for i in range(nclusters)]
    twoSigma = [[] for i in range(nclusters)]
    outliers = [[] for i in range(nclusters)]
    distFromCenter = [[] for i in range(nclusters)]
    
    allOutliers=[]
    for i in range(nclusters):
        # need to keep track of which points get pulled into each cluster:
        clusterIndexes[i]=[j for j in range(len(data)) if clusters[j]==i]
        # below originally in same form as clusterIndexes, should probably change back
        for j in range(len(clusterIndexes[i])):
            cluster[i].append(data[clusterIndexes[i][j]])

        distFromCenter = [0 for k in range(len(cluster[i]))] # reinitializing this array
        
        """
        ========== Checking density of clusters ============
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
        ========== Finding points outside of 2-sigma ===========
        """
        
        twoSigma[i]=(2*np.std(cluster[i]))
        
        # Calculate distances to each point in each cluster to the center of its cluster
        for j in range(len(cluster[i])):
            centerloc=centers[i] # center of the cluster
            dataloc=cluster[i][j] # location of the datapoint
            sqrd=0
            for x in range(len(centers)):
                sqrd+=pow(dataloc[x]-centerloc[x],2) # (x-x0)^2+(y-y0)^2+...
            distance = pow(sqrd,0.5) # sqrt((x-x0)^2+(y-y0)^2+...)
            distFromCenter[j]=distance 

        # cluster i, lightcurve j, feature k
        #distFromCenter[i]=[distance(center[i],cluster[i][j]) for j in range(len(cluster))]
        #outliers=[k for k in range(len(cluster[i])) if distFromCenter[k]>twoSigma[i]]
        outliers=[k for k in range(len(cluster[i])) if distFromCenter[k]>=.5*max(distFromCenter)]
        
        # place outliers from this cluster into general outlier list
        if len(outliers)==0:
            print("none")
        else:
            for k in outliers:
                allOutliers.append(clusterIndexes[i][k])      
        
    allOutliers.sort()
        
    return clusters,allOutliers

def plot_fit(ffeatures,clusters):
    # Set the dictionaries that contain labels and data. Radiobuttons will have labels from listoffeatures
    # Betterlabels will contain the titles of axes we'll actually want on there
    listoffeatures = ['longtermtrend', 'meanmedrat', 'skews', 'varss', 'coeffvar', 'stds', 'numoutliers', 'numnegoutliers', 'numposoutliers', 'numout1s', 'kurt', 'mad', 'maxslope', 'minslope', 'meanpslope', 'meannslope', 'g_asymm', 'rough_g_asymm', 'diff_asymm', 'skewslope', 'varabsslope', 'varslope', 'meanabsslope', 'absmeansecder', 'num_pspikes', 'num_nspikes', 'num_psdspikes', 'num_nsdspikes','stdratio', 'pstrend', 'num_zcross', 'num_pm', 'len_nmax', 'len_nmin', 'mautocorrcoef', 'ptpslopes', 'periodicity', 'periodicityr', 'naiveperiod', 'maxvars', 'maxvarsr', 'oeratio', 'amp', 'normamp','mbp', 'mid20', 'mid35', 'mid50', 'mid65', 'mid80', 'percentamp', 'magratio', 'sautocorrcoef', 'autocorrcoef', 'flatmean', 'tflatmean', 'roundmean', 'troundmean', 'roundrat', 'flatrat']
    
    betterlabels = ['longtermtrend', 'meanmedrat', 'skews', 'varss', 'coeffvar', 'stds', 'numoutliers', 'numnegoutliers', 'numposoutliers', 'numout1s', 'kurt', 'mad', 'maxslope', 'minslope', 'meanpslope', 'meannslope', 'g_asymm', 'rough_g_asymm', 'diff_asymm', 'skewslope', 'varabsslope', 'varslope', 'meanabsslope', 'absmeansecder', 'num_pspikes', 'Number of Negative Spikes (Slope > 3*sigma)', 'num_psdspikes', 'num_nsdspikes','stdratio', 'pstrend', 'Number of Longterm Trendline Crossings', 'Number of Peaks', 'len_nmax', 'len_nmin', 'mautocorrcoef', 'ptpslopes', 'periodicity', 'periodicityr', 'naiveperiod', 'maxvars', 'maxvarsr', 'oeratio', 'amp', 'normamp','mbp', 'mid20', 'mid35', 'mid50', 'mid65', 'mid80', 'percentamp', 'magratio', 'sautocorrcoef', 'autocorrcoef', 'flatmean', 'tflatmean', 'roundmean', 'troundmean', 'roundrat', 'flatrat']
    
    # labeldict connects the variable's code name in listoffeatures to it's presentable name in betterlabels
    
    labeldict= dict(zip(listoffeatures,betterlabels))
    # datadict connects label to data
    datadict = dict(zip(listoffeatures,ffeatures))
    
    # labellist keeps track of what each axis should be labelled, initialized with first 3 features
    labellist=[labeldict[listoffeatures[30]],labeldict[listoffeatures[31]],labeldict[listoffeatures[25]]]
    
    # axesdict could probably be accomoplished with a list, 
    # I'm partial to the dictionary because it helps me keep things associated directly.
    
    axesdict = {'xaxis':ffeatures[30],'yaxis':ffeatures[31],'zaxis':ffeatures[25]}
    
    # Plot initializing stuff
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(axesdict['xaxis'], axesdict['yaxis'], axesdict['zaxis'], c=clusters.astype(np.float))
    plt.subplots_adjust(left=0.3)
    ax.set_xlabel(labellist[0])
    ax.set_ylabel(labellist[1])
    ax.set_zlabel(labellist[2])

    """
    Matplotlib radiobuttons format poorly when there are more than ~5 buttons, this is an issue that will be resolved at some point in the future.
    For the time being, I'm making a bunch of radiobutton sections with 5 buttons apeice. Each column represents a differnt axis.
    The most recently clicked button will be the active one, so the color coding is less than useful.

    It's not pretty but it works.
    """

    rax1p = [[0.0,.92-i/12.0,0.1,1.0/12.0] for i in range(12)]
    
    rax10,rax11,rax12,rax13,rax14,rax15,rax16,rax17,rax18,rax19,rax110,rax111 = plt.axes(rax1p[0]), plt.axes(rax1p[1]), plt.axes(rax1p[2]), plt.axes(rax1p[3]), plt.axes(rax1p[4]), plt.axes(rax1p[5]), plt.axes(rax1p[6]), plt.axes(rax1p[7]), plt.axes(rax1p[8]), plt.axes(rax1p[9]), plt.axes(rax1p[10]), plt.axes(rax1p[11])
    
    radio10,radio11,radio12,radio13,radio14,radio15,radio16,radio17,radio18,radio19,radio110,radio111 = RadioButtons(rax10, listoffeatures[0:5]),RadioButtons(rax11, listoffeatures[5:10]),RadioButtons(rax12, listoffeatures[10:15]),RadioButtons(rax13, listoffeatures[15:20]),RadioButtons(rax14, listoffeatures[20:25]),RadioButtons(rax15, listoffeatures[25:30]),RadioButtons(rax16, listoffeatures[30:35]),RadioButtons(rax17, listoffeatures[35:40]),RadioButtons(rax18, listoffeatures[40:45]),RadioButtons(rax19, listoffeatures[45:50]),RadioButtons(rax110, listoffeatures[50:55]),RadioButtons(rax111, listoffeatures[55:])
    
    def axis1(label):
        ax.cla()
        axesdict['xaxis'] = [datadict[label]]
        ax.scatter(axesdict['xaxis'], axesdict['yaxis'], axesdict['zaxis'], c=clusters.astype(np.float))
        labellist[0] = labeldict[label]
        ax.set_xlabel(labellist[0])
        ax.set_ylabel(labellist[1])
        ax.set_zlabel(labellist[2])
        plt.draw()

    radio10.on_clicked(axis1)
    radio11.on_clicked(axis1)
    radio12.on_clicked(axis1)
    radio13.on_clicked(axis1)
    radio14.on_clicked(axis1)
    radio15.on_clicked(axis1)
    radio16.on_clicked(axis1)
    radio17.on_clicked(axis1)
    radio18.on_clicked(axis1)
    radio19.on_clicked(axis1)
    radio110.on_clicked(axis1)
    radio111.on_clicked(axis1)


    rax2p = [[0.1,.92-i/12.0,0.1,1.0/12.0] for i in range(12)]

    rax20,rax21,rax22,rax23,rax24,rax25,rax26,rax27,rax28,rax29,rax210,rax211 = plt.axes(rax2p[0]), plt.axes(rax2p[1]), plt.axes(rax2p[2]), plt.axes(rax2p[3]), plt.axes(rax2p[4]), plt.axes(rax2p[5]), plt.axes(rax2p[6]), plt.axes(rax2p[7]), plt.axes(rax2p[8]), plt.axes(rax2p[9]), plt.axes(rax2p[10]), plt.axes(rax2p[11])

    radio20, radio21, radio22, radio23, radio24, radio25, radio26, radio27, radio28, radio29, radio210, radio211 = RadioButtons(rax20, listoffeatures[0:5]), RadioButtons(rax21, listoffeatures[5:10]), RadioButtons(rax22, listoffeatures[10:15]), RadioButtons(rax23, listoffeatures[15:20]), RadioButtons(rax24, listoffeatures[20:25]), RadioButtons(rax25, listoffeatures[25:30]), RadioButtons(rax26, listoffeatures[30:35]), RadioButtons(rax27, listoffeatures[35:40]), RadioButtons(rax28, listoffeatures[40:45]), RadioButtons(rax29, listoffeatures[45:50]), RadioButtons(rax210, listoffeatures[50:55]), RadioButtons(rax211, listoffeatures[55:])
    
    def axis2(label):
        ax.cla()
        axesdict['yaxis'] = [datadict[label]]
        ax.scatter(axesdict['xaxis'], axesdict['yaxis'], axesdict['zaxis'], c=clusters.astype(np.float))
        labellist[1] = label
        ax.set_xlabel(labeldict[labellist[0]])
        ax.set_ylabel(labeldict[labellist[1]])
        ax.set_zlabel(labeldict[labellist[2]])
        plt.draw()

    radio20.on_clicked(axis2)
    radio21.on_clicked(axis2)
    radio22.on_clicked(axis2)
    radio23.on_clicked(axis2)
    radio24.on_clicked(axis2)
    radio25.on_clicked(axis2)
    radio26.on_clicked(axis2)
    radio27.on_clicked(axis2)
    radio28.on_clicked(axis2)
    radio29.on_clicked(axis2)
    radio210.on_clicked(axis2)
    radio211.on_clicked(axis2)

    rax3p = [[0.2,.92-i/12.0,0.1,1.0/12.0] for i in range(12)]

    rax30,rax31,rax32,rax33,rax34,rax35,rax36,rax37,rax38,rax39,rax310,rax311 = plt.axes(rax3p[0]), plt.axes(rax3p[1]), plt.axes(rax3p[2]), plt.axes(rax3p[3]), plt.axes(rax3p[4]), plt.axes(rax3p[5]), plt.axes(rax3p[6]), plt.axes(rax3p[7]), plt.axes(rax3p[8]), plt.axes(rax3p[9]), plt.axes(rax3p[10]), plt.axes(rax3p[11])

    radio30, radio31, radio32, radio33, radio34, radio35, radio36, radio37, radio38, radio39, radio310, radio311 = RadioButtons(rax30, listoffeatures[0:5]), RadioButtons(rax31, listoffeatures[5:10]), RadioButtons(rax32, listoffeatures[10:15]), RadioButtons(rax33, listoffeatures[15:20]), RadioButtons(rax34, listoffeatures[20:25]), RadioButtons(rax35, listoffeatures[25:30]), RadioButtons(rax36, listoffeatures[30:35]), RadioButtons(rax37, listoffeatures[35:40]), RadioButtons(rax38, listoffeatures[40:45]), RadioButtons(rax39, listoffeatures[45:50]), RadioButtons(rax310, listoffeatures[50:55]), RadioButtons(rax311, listoffeatures[55:])

    def axis3(label):
        ax.cla()
        axesdict['zaxis'] = [datadict[label]]
        ax.scatter(axesdict['xaxis'], axesdict['yaxis'], axesdict['zaxis'], c=clusters.astype(np.float))
        labellist[2] = label
        ax.set_xlabel(labeldict[labellist[0]])
        ax.set_ylabel(labeldict[labellist[1]])
        ax.set_zlabel(labeldict[labellist[2]])
        plt.draw()
        
    radio30.on_clicked(axis3)
    radio31.on_clicked(axis3)
    radio32.on_clicked(axis3)
    radio33.on_clicked(axis3)
    radio34.on_clicked(axis3)
    radio35.on_clicked(axis3)
    radio36.on_clicked(axis3)
    radio37.on_clicked(axis3)
    radio38.on_clicked(axis3)
    radio39.on_clicked(axis3)
    radio310.on_clicked(axis3)
    radio311.on_clicked(axis3)

    plt.show()

ffeaturesID = str(identifier)+'dataByFeature.npy'
ffeatures = np.load(ffeaturesID)
dataID = str(identifier)+'dataByLightCurve.npy'
data = np.load(dataID)
filelist = str(identifier)+'filelist'
files = [line.strip() for line in open(filelist)]

clusterlabels,outliers=KMeans_clusters(data,nclusters)

for i in outliers:
    print files[i]

plot_fit(ffeatures,clusterlabels)
