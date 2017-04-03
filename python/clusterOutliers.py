import numpy as np
from multiprocessing import Pool,cpu_count
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from matplotlib import colors
import matplotlib.cm as cmx

import keplerml
import km_outliers
import db_outliers

class clusterOutliers(object):
    def __init__(self,feats,fitsDir):
        self.data = pd.read_csv(feats,index_col=0)
        self.fitsDir = fitsDir
        self.files = self.data.index
        # Initializing the data and files samples with the first 100 entries.
        self.dataSample = self.data.iloc[:100]
        self.filesSample = self.files[:100]
        self.lcs = self.poolRKC(self.filesSample)
        self.importedForPlotting = False
        self.sampleGenerated = False
        self.sampleTSNE = False
        
    def poolRKC(self,files):
        # Read in kepler curves in parallel. 
        # Returns array containing of arrays of [t,nf,err] for each lc.
        numcpus = cpu_count()
        fwp = self.fitsDir+"/"+files
        p = Pool(numcpus)
        lcs = p.map(keplerml.read_kepler_curve,fwp)
        p.close()
        p.join
        print("Done.")
        return lcs
    
    def randSample(self, numLCs):
        """
        Returns a random sample of numLCs light curves, data returned as an array
        of shape [numLCs,3,len(t)]
        Rerunning this, or randSampleWTabby will replace the previous random sample.
        """
        assert (numLCs <self.files.size),"Number of samples greater than the number of files."
        self.numLCs = numLCs
        print("Creating random file list...")
        self.dataSample = self.data.sample(n=numLCs)
        self.filesSample = self.dataSample.index
        print("Importing lightcurves...")
        self.lcs = self.poolRKC(self.filesSample)
        self.sampleGenerated = True
        return self.lcs
    
    def randSampleWTabby(self, numLCs):
        """
        Returns a random sample of numLCs light curves, data returned as an array
        of shape [numLCs,3,len(t)]
        Rerunning this, or randSample will replace the previous random sample.
        """
        assert (numLCs < len(self.filesSample)),"Number of samples greater than the number of files."
        self.numLCs = numLCs
        print("Creating random file list...")
        self.dataSample = self.data.sample(n=numLCs)

        print("Checking for Tabby...")
        if not dataSample.index.str.contains('8462852').all():
            print("Adding Tabby...")
            self.dataSample.drop(self.dataSample.index[0])
            self.dataSample.append(self.data[self.data.index.str.contains('8462852')])
        self.filesSample = self.dataSample.index
        print("Importing lightcurves...")
        self.lcs = self.poolRKC(self.filesSample)
        self.sampleGenerated = True
        return self.lcs
    
    def fullQ(self):
        self.filesSample = self.files
        self.dataSample = self.data
        self.lcs = self.poolRKC(self.filesSample)
        return self.lcs
    
    def sample_tsne_fit(self):
        """
        Performs a t-SNE dimensionality reduction on the data sample generated.
        Uses a PCA initialization and the perplexity given, or defaults to 50.
        
        Appends the dataSample dataframe with the t-SNE X and Y coordinates
        Returns tsneX and tsneY
        """
        assert self.sampleGenerated,"Sample has not yet been generated using randSample or randSampleWTabby"
        perplexity=50
        scaler = preprocessing.StandardScaler().fit(self.dataSample)
        scaledData = scaler.transform(self.dataSample)
        tsne = TSNE(n_components=2,perplexity=perplexity,init='random',verbose=True)
        tsne_fit=tsne.fit_transform(scaledData)
        self.dataSample['tsne_x'] = tsne_fit.T[0]
        self.dataSample['tsne_y'] = tsne_fit.T[1]
        # Goal is to minimize the KL-Divergence
        if sklearn.__version__ == '0.18.1':
            print("KL-Divergence was %s"%tsne.kl_divergence_ )
        print("Done.")
        self.sampleTSNE = True
        return
    
    def tsne_fit(self,data):
        """
        Performs a t-SNE dimensionality reduction on the data sample generated.
        Uses a PCA initialization and the perplexity given, or defaults to 50.
        
        Appends the dataSample dataframe with the t-SNE X and Y coordinates
        Returns tsneX and tsneY
        """
        
        perplexity=50
        scaler = preprocessing.StandardScaler().fit(data)
        scaledData = scaler.transform(data)
        tsne = TSNE(n_components=2,perplexity=perplexity,init='pca',verbose=True)
        fit=tsne.fit_transform(scaledData)
        # Goal is to minimize the KL-Divergence
        if sklearn.__version__ == '0.18.1':
            print("KL-Divergence was %s"%tsne.kl_divergence_ )
        print("Done.")
        return fit
    
    def sample_km_out(self):
        assert self.sampleTSNE,"Sample has not been reduced with sample_tsne_fit yet."
        tsneData = self.dataSample[['tsne_x','tsne_y']]
        clusterLabels = km_outliers.kmeans_w_outliers(tsneData,1)
        self.dataSample['km_cluster']=clusterLabels
        return
    
    def km_out(self,df):
        clusterLabels = km_outliers.kmeans_w_outliers(df,1)
        return clusterLabels
        
    def sample_db_out(self):
        assert self.sampleTSNE,"Sample has not been reduced with sample_tsne_fit yet."
        clusterLabels = db_outliers.dbscan_w_outliers(self.dataSample[['tsne_x','tsne_y']])
        self.dataSample['db_cluster']=clusterLabels
        return
    
    def db_out(self,df):
        clusterLabels = db_outliers.dbscan_w_outliers(df)
        return clusterLabels
        
    def import_for_plot(self,method):
        self.importedForPlotting = True
        """--- import light curve data ---"""
        pathtofits = self.fitsDir

        # The following needs to be generated in the cell above.
        files = self.filesSample
        if method == 'dbscan':
            clusterLabels = self.dataSample.db_cluster
        elif method == 'kmeans':
            clusterLabels = self.dataSample.km_cluster
        # data is an array containing each data point
        data = np.array([np.array(self.dataSample[['tsne_x','tsne_y']].loc[i]) for i in self.dataSample.index])

        cNorm  = colors.Normalize(vmin=0, vmax=max(clusterLabels))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='jet')

        # tsneX has all the x-coordinates
        tsneX = data.T[0]
        # tsneY has all the y-coordinates
        tsneY = data.T[1]   
        outX = []
        outY = []
        files_out = []
        clusterX = []
        clusterY = []
        files_cluster = []

        for i in enumerate(data):
            if clusterLabels[i[0]] == -1:
                outX.append(i[1][0])
                outY.append(i[1][1])
                files_out.append(files[i[0]])
            else:
                clusterX.append(i[1][0])
                clusterY.append(i[1][1])
                files_cluster.append(files[i[0]])

        lightcurveData = self.lcs

        """--- Organizing data and Labels ---"""


        if self.dataSample.index.str.contains('8462852').any():
            tabbyInd = list(self.dataSample.index).index('8462852')            
        else:
            tabbyInd = 0
        
        return files, clusterLabels, data, cNorm, scalarMap, tsneX, tsneY, outX, outY, files_out, clusterX, clusterY, files_cluster, lightcurveData, tabbyInd
    
    def save(self,of):
        data.to_csv(of)