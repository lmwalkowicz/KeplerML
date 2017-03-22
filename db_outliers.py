"""
DBSCAN Clustering
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import fnmatch


def eps_est(data):
    # distance array containing all distances
    nbrs = NearestNeighbors(n_neighbors=np.ceil(.2*len(data)), algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    # Distance to 200th nearest neighbor, using 200th instead of 4th because: ... reasons
    distArr = distances[:,np.ceil(.02*len(data))]
    distArr.sort()
    pts = range(len(distArr))

    # The following looks for the first instance (past the mid point) where the mean of the following 50 points
    # is at least 5% greater than the mean of the previous 50 points.
    # Alternatively, perhaps a better method, we could consider the variance of the points and draw conclusions from that
    if len(data) <= 200:
        number = 10
    else:
        number = 50
    cutoff = 1.05
    for i in range(number,len(pts)-number):
        if np.mean(distArr[i+1:i+number])>=cutoff*np.mean(distArr[i-number:i-1]) and i>(len(pts)%2+len(pts))/2:

            dbEps = distArr[i]
            break

    # Estimating nneighbors by finding the number of pts. 
    # that fall w/in our determined eps for each point.

    count = np.zeros(len(pts))
    for i in pts:    
        for dist in distances[i]:
            if dist <= dbEps:
                count[i]+=1
    average = np.median(count)
    sigma = np.std(count)
    neighbors = average/2 # Divide by 2 for pts on the edges
    print("""
    Epsilon is in the neighborhood of %s, 
    with an average of %s neighbors within epsilon,
    %s neighbors in half circle (neighbors/2).
    """%(dbEps,average,neighbors))
    return dbEps,neighbors

def dbscan_w_outliers(data):
    X=np.array([np.array(data.loc[i]) for i in data.index])
    print("Clustering data...")
    dbEps,neighbors= eps_est(X)
    
    print("Clustering data with DBSCAN...")
    est = DBSCAN(eps=dbEps,min_samples=neighbors)
    est.fit(X)
    clusterLabels = est.labels_

    numout = len(clusterLabels[clusterLabels==-1])
    numclusters = max(clusterLabels+1)
    if data.index.str.contains('8462852').any():
        tabbyInd = list(data.index).index('8462852')
        if clusterLabels[tabbyInd] == -1:
            print("Tabby has been found to be an outlier in DBSCAN.")
        else:
            print("Tabby has not Not found to be an outlier in DBSCAN")
        
    print("There were %s clusters and %s total outliers"%(numclusters,numout))

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
        print("No output path specified, saving to 'output.npy' in local folder.")
        of = 'output'
    
    np.save(of,dbscan_w_outliers(df.index,df[['tsne_x','tsne_y']]))
    print("Done.")