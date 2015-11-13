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



listoffeatures = ['longtermtrend', 'meanmedrat', 'skews', 'varss', 'coeffvar', 'stds', 'numoutliers', 'numnegoutliers', 'numposoutliers', 'numout1s', 'kurt', 'mad', 'maxslope', 'minslope', 'meanpslope', 'meannslope', 'g_asymm', 'rough_g_asymm', 'diff_asymm', 'skewslope', 'varabsslope', 'varslope', 'meanabsslope', 'absmeansecder', 'num_pspikes', 'num_nspikes', 'num_psdspikes', 'num_nsdspikes','stdratio', 'pstrend', 'num_zcross', 'num_pm', 'len_nmax', 'len_nmin', 'mautocorrcoef', 'ptpslopes', 'periodicity', 'periodicityr', 'naiveperiod', 'maxvars', 'maxvarsr', 'oeratio', 'amp', 'normamp','mbp', 'mid20', 'mid35', 'mid50', 'mid65', 'mid80', 'percentamp', 'magratio', 'sautocorrcoef', 'autocorrcoef', 'flatmean', 'tflatmean', 'roundmean', 'troundmean', 'roundrat', 'flatrat']

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

def calc_outliers_pts(t, nf):
    # Is t really a necessary input? The answer is no, but eh, why change it
    posthreshold = np.mean(nf)+4*np.std(nf)
    negthreshold = np.mean(nf)-4*np.std(nf)
    
    numposoutliers,numnegoutliers,numout1s=0,0,0
    for j in range(len(nf)):
        # First checks if nf[j] is outside of 1 sigma
        if abs(np.mean(nf)-nf[j])>np.std(nf):
            numout1s += 1
            if nf[j]>posthreshold:
                numposoutliers += 1
            elif nf[j]<negthreshold:
                numnegoutliers += 1
    numoutliers=numposoutliers+numnegoutliers
    
    return numoutliers, numposoutliers, numnegoutliers, numout1s


def calc_slopes(t, nf, corrnf):

    slope_array = np.zeros(20)

    #Delta flux/ Delta time
    slopes=[(nf[j+1]-nf[j])/(t[j+1]-t[j]) for j in range (len(nf)-1)]
    #corrslopes removes the longterm linear trend (if any) and then looks at the slope
    corrslopes=[(corrnf[j+1]-corrnf[j])/(t[j+1]-t[j]) for j in range (len(corrnf)-1)]
    meanslope = np.mean(slopes)
    # by looking at where the 99th percentile is instead of just the largest number,
    # I think it avoids the extremes which might not be relevant (might be unreliable data)
    # Is the miniumum slope the most negative one, or the flattest one? Most negative
    maxslope=np.percentile(slopes,99)
    minslope=np.percentile(slopes,1)
    # Separating positive slopes and negative slopes
    # Should both include the 0 slope? I'd guess there wouldn't be a ton, but still...
    pslope=[slopes[j] for j in range(len(slopes)) if slopes[j]>=0]
    nslope=[slopes[j] for j in range(len(slopes)) if slopes[j]<=0]
    # Looking at the average (mean) positive and negative slopes
    meanpslope=np.mean(pslope)
    meannslope=np.mean(nslope)
    # Quantifying the difference in shape.
    g_asymm=meanpslope / meannslope
    # Won't this be skewed by the fact that both pslope and nslope have all the 0's? Eh
    rough_g_asymm=len(pslope) / len(nslope)
    # meannslope is inherently negative, so this is the difference btw the 2
    diff_asymm=meanpslope + meannslope
    skewslope = scipy.stats.skew(slopes)
    absslopes=[abs(slopes[j]) for j in range(len(slopes))]
    meanabsslope=np.mean(absslopes)
    varabsslope=np.var(absslopes)
    varslope=np.var(slopes)
    #secder = Second Derivative
    # Reminder for self: the slope is "located" halfway between the flux and time points, 
    # so the delta t in the denominator is accounting for that.
    #secder=[(slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2) for j in range(1, len(nf)-1)]
    #algebraic simplification:
    secder=[2*(slopes[j]-slopes[j-1])/(t[j+1]-t[j-1]) for j in range(1, len(nf)-1)]
    meansecder=np.mean(secder)
    #abssecder=[abs((slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2)) for j in range (1, len(slopes)-1)]
    # simplification:
    abssecder=[abs(secder[j]) for j in range(1, len(secder))]
    absmeansecder=np.mean(abssecder)

    pslopestds=np.std(pslope)
    nslopestds=np.std(nslope)
    sdstds=np.std(secder)
    meanstds=np.mean(secder)
    stdratio=pslopestds/nslopestds

    pspikes =[slopes[j] for j in range(len(slopes)) if slopes[j]>=meanpslope+3*pslopestds] 
    nspikes=[slopes[j] for j in range(len(slopes)) if slopes[j]<=meannslope-3*nslopestds]
    psdspikes=[secder[j] for j in range(len(secder)) if secder[j]>=4*sdstds] 
    nsdspikes=[secder[j] for j in range(len(secder)) if secder[j]<=-4*sdstds]

    num_pspikes = len(pspikes)
    num_nspikes = len(nspikes)
    num_psdspikes = len(psdspikes)
    num_nsdspikes = len(nsdspikes)
    
    stdratio = pslopestds / nslopestds
    # The ratio of postive slopes with a following postive slope to the total number of points.

    pstrend=len([slopes[j] for j in range(len(slopes)-1) if (slopes[j]>0) & (slopes[j+1]>0)])/len(slopes)

    slope_array = [meanslope, maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes, stdratio, pstrend]

    return slopes, corrslopes, secder, slope_array

def calc_maxmin_periodics(t, nf, err):
#look up this heapq.nlargest crap
    #This looks up the local maximums. Adds a peak if it's the largest within 10 points on either side.

    naivemax,nmax_times = [],[]
    naivemins = []
    for j in range(len(nf)):
        if nf[j] == max(nf[max(j-10,0):min(j+10,len(nf-1))]):
            naivemax.append(nf[j])
            nmax_times.append(t[j])
        elif nf[j] == min(nf[max(j-10,0):min(j+10,len(nf-1))]):
            naivemins.append(nf[j])
    len_nmax=len(naivemax) #F33
    len_nmin=len(naivemins) #F34
    

    #wtf is this?
    #D: shifts everything to the left for some reason.
    #autopdcmax = [naivemax[j+1] for j in range(len(naivemax)-1)] = naivemax[1:]
    
    #naivemax[:-1:] is naivemax without the last value and autopdcmax is naivemax without the first value. why do this?a
    #np.corrcoef(array) returns a correlation coefficient matrix. I.e. a normalized covariance matrix
    """
    It looks like it compares each point to it's next neighbor, hence why they're offset, 
    then determines if there's a correlation between the two. If the coefficient is closer
    to 1, then there's a strong correlation, if 0 then no correlation, if -1 (possible?) then anti-correlated.
    """
    mautocorrcoef = np.corrcoef(naivemax[:-1], naivemax[1:])[0][1] #F35
    mautocovs = np.cov(naivemax[:-1],naivemax[1:])[0][1] # Not a feature, not used elsewhere

    """peak to peak slopes"""
    ppslopes = [abs((naivemax[j+1]-naivemax[j])/(nmax_times[j+1]-nmax_times[j])) for j in range(len(naivemax)-1)]

    ptpslopes=np.mean(ppslopes) #F36

    maxdiff=[nmax_times[j+1]-nmax_times[j] for j in range(len(naivemax)-1)]

    periodicity=np.std(maxdiff)/np.mean(maxdiff) #F37
    periodicityr=np.sum(abs(maxdiff-np.mean(maxdiff)))/np.mean(maxdiff) #F38

    naiveperiod=np.mean(maxdiff) #F39
    # why not maxvars = np.var(naivemax)? Is this not the variance? Seems like it should be...
    #maxvars = np.var(naivemax) #F40
    maxvars = np.std(naivemax)/np.mean(naivemax) #F40
    # I don't understand what this is.
    maxvarsr = np.sum(abs(naivemax-np.mean(naivemax)))/np.mean(naivemax) #F41

    emin = naivemins[::2] # even indice minimums
    omin = naivemins[1::2] # odd indice minimums
    meanemin = np.mean(emin)
    meanomin = np.mean(omin)
    oeratio = meanomin/meanemin #F42

    peaktopeak_array = [len_nmax, len_nmin, mautocorrcoef, ptpslopes, periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio]

    return peaktopeak_array, naivemax, naivemins



def lc_examine(filelist, style='-'):
    """Takes a list of FITS files and plots them one by one sequentially"""
    files = [line.strip() for line in open(filelist)]

    for i in range(len(files)):
        lc = pyfits.getdata(files[i])
        t = lc.field('TIME')
        f = lc.field('PDCSAP_FLUX')
        nf = f / np.median(f)
 
        title = 'Light curve for {0}'.format(files[i])
        plt.plot(t, nf, style)
        plt.title(title)
        plt.xlabel(r'$t (days)$')
        plt.ylabel(r'$\Delta F$')
        plt.show()

    return
def fcalc(nfile):
    # Keeping track of progress, noting every thousand files completed.

    t,nf,err = read_kepler_curve(nfile)

    # t = time
    # err = error
    # nf = normalized flux. Same as mf but offset by 1 to center at 0?

    longtermtrend = np.polyfit(t, nf, 1)[0] # Feature 1 (Abbr. F1) overall slope
    yoff = np.polyfit(t, nf, 1)[1] # Not a feature? y-intercept of linear fit
    meanmedrat = np.mean(nf) / np.median(nf) # F2
    skews = scipy.stats.skew(nf) # F3
    varss = np.var(nf) # F4
    coeffvar = np.std(nf)/np.mean(nf) #F5
    stds = np.std(nf) #F6

    corrnf = nf - longtermtrend*t - yoff #this removes any linear slope to lc so you can look at just troughs - is this a sign err tho?
    # D: I don't think there's a sign error

    # Features 7 to 10
    numoutliers, numposoutliers, numnegoutliers, numout1s = calc_outliers_pts(t, nf)

    kurt = scipy.stats.kurtosis(nf)

    mad = np.median([abs(nf[j]-np.median(nf)) for j in range(len(nf))])

    # slopes array contains features 13-30
    slopes, corrslopes, secder, slopes_array = calc_slopes(t, nf, corrnf) 

    maxslope = slopes_array[0]
    minslope = slopes_array[1]
    meanpslope  = slopes_array[2]
    meannslope  = slopes_array[3]
    g_asymm = slopes_array[4]
    rough_g_asymm  = slopes_array[5]
    diff_asymm  = slopes_array[6]
    skewslope  = slopes_array[7]
    varabsslope  = slopes_array[8]
    varslope  = slopes_array[9]
    meanabsslope  = slopes_array[10]
    absmeansecder = slopes_array[11]
    num_pspikes = slopes_array[12]
    num_nspikes  = slopes_array[13]
    num_psdspikes = slopes_array[14]
    num_nsdspikes = slopes_array[15]
    stdratio = slopes_array[16]
    pstrend = slopes_array[17]

    # Checks if the flux crosses the zero line.
    zcrossind= [j for j in range(len(nf)-1) if corrnf[j]*corrnf[j+1]<0]
    num_zcross = len(zcrossind) #F31

    plusminus=[j for j in range(1,len(slopes)) if (slopes[j]<0)&(slopes[j-1]>0)]
    num_pm = len(plusminus)

    # peak to peak array contains features 33 - 42
    peaktopeak_array, naivemax, naivemins = calc_maxmin_periodics(t, nf, err)

    len_nmax=peaktopeak_array[0]
    len_nmin=peaktopeak_array[1]
    mautocorrcoef=peaktopeak_array[2]
    ptpslopes=peaktopeak_array[3]
    periodicity=peaktopeak_array[4]
    periodicityr=peaktopeak_array[5]
    naiveperiod=peaktopeak_array[6]
    maxvars=peaktopeak_array[7]
    maxvarsr=peaktopeak_array[8]
    oeratio=peaktopeak_array[9]

    # amp here is actually amp_2 in revantese
    # 2x the amplitude (peak-to-peak really), the 1st percentile will be negative, so it's really adding magnitudes
    amp = np.percentile(nf,99)-np.percentile(nf,1) #F43
    normamp = amp / np.mean(nf) #this should prob go, since flux is norm'd #F44

    # ratio of points within 10% of middle to total number of points 
    mbp = len([nf[j] for j in range(len(nf)) if (nf[j] < (np.median(nf) + 0.1*amp)) & (nf[j] > (np.median(nf)-0.1*amp))]) / len(nf) #F45

    f595 = np.percentile(nf,95)-np.percentile(nf,5)
    f1090 =np.percentile(nf,90)-np.percentile(nf,10)
    f1782 =np.percentile(nf, 82)-np.percentile(nf, 17)
    f2575 =np.percentile(nf, 75)-np.percentile(nf, 25)
    f3267 =np.percentile(nf, 67)-np.percentile(nf, 32)
    f4060 =np.percentile(nf, 60)-np.percentile(nf, 40)
    mid20 =f4060/f595 #F46
    mid35 =f3267/f595 #F47
    mid50 =f2575/f595 #F48
    mid65 =f1782/f595 #F49
    mid80 =f1090/f595 #F50 

    percentamp = max([(nf[j]-np.median(nf)) / np.median(nf) for j in range(len(nf))]) #F51
    magratio = (max(nf)-np.median(nf)) / amp #F52

    #autopdc=[nf[j+1] for j in range(len(nf)-1)] = nf[1:]
    autocorrcoef = np.corrcoef(nf[:-1], nf[1:])[0][1] #F54
    #autocovs = np.cov(nf[:-1], nf[1:])[0][1] # not used for anything...

    #sautopdc=[slopes[j+1] for j in range(len(slopes)-1)] = slopes[1:]

    sautocorrcoef = np.corrcoef(slopes[:-1], slopes[1:])[0][1] #F55
    #sautocovs = np.cov(slopes[:-1:],slopes[1:])[0][1] # not used for anything...

    flatness = [np.mean(slopes[max(0,j-6):min(j-1, len(slopes)-1):1])- np.mean(slopes[max(0,j):min(j+4, len(slopes)-1):1]) for j in range(len(slopes)) if nf[j] in naivemax]

    flatmean = np.nansum(flatness)/len(flatness) #F55

    # trying flatness w slopes and nf rather than "corr" vals, despite orig def in RN's program
    tflatness = [-np.mean(slopes[max(0,j-6):min(j-1, len(slopes)-1):1])+ np.mean(slopes[max(0,j):min(j+4, len(slopes)-1):1]) for j in range(len(slopes)) if nf[j] in naivemins] 
    # tflatness for mins, flatness for maxes
    tflatmean = np.nansum(tflatness) / len(tflatness) #F56

    roundness=[np.mean(secder[max(0,j-6):min(j-1, len(secder)-1):1]) + np.mean(secder[max(0,j+1):min(j+6, len(secder)-1):1]) for j in range(len(secder)) if nf[j+1] in naivemax]

    roundmean = np.nansum(roundness) / len(roundness) #F57

    troundness = [np.mean(secder[max(0,j-6):min(j-1, len(secder)-1):1]) + np.mean(secder[max(0,j+1):min(j+6, len(secder)-1):1]) for j in range(len(secder)) if nf[j+1] in naivemins]

    troundmean = np.nansum(troundness)/len(troundness) #F58
    roundrat = roundmean / troundmean #F59

    flatrat = flatmean / tflatmean #F60

    return longtermtrend, meanmedrat, skews, varss, coeffvar, stds, numoutliers, numnegoutliers, numposoutliers, numout1s, kurt, mad, maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes,stdratio, pstrend, num_zcross, num_pm, len_nmax, len_nmin, mautocorrcoef, ptpslopes, periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio, amp, normamp,mbp, mid20, mid35, mid50, mid65, mid80, percentamp, magratio, sautocorrcoef, autocorrcoef, flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat

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
    xmin = [min(X,key=lambda a:a[i])[i] for i in range(60)]
    xmax = [max(X,key=lambda a:a[i])[i] for i in range(60)]
    return (xmin,xmax)

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
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin[j],xmax[j]) for j in range(60)])
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

def kmeans3D(ffeatures,data):
    # Run KMeans, get clusters
    npdata = np.array(data)
    est = KMeans(n_clusters=optimalK(npdata))
    est.fit(npdata)
    labels = est.labels_
    
    # Set the dictionaries that contain labels and data. Radiobuttons will have labels from listoffeatures
    # Betterlabels will contain the titles of axes we'll actually want on there
    listoffeatures = ['longtermtrend', 'meanmedrat', 'skews', 'varss', 'coeffvar', 'stds', 'numoutliers', 'numnegoutliers', 'numposoutliers', 'numout1s', 'kurt', 'mad', 'maxslope', 'minslope', 'meanpslope', 'meannslope', 'g_asymm', 'rough_g_asymm', 'diff_asymm', 'skewslope', 'varabsslope', 'varslope', 'meanabsslope', 'absmeansecder', 'num_pspikes', 'num_nspikes', 'num_psdspikes', 'num_nsdspikes','stdratio', 'pstrend', 'num_zcross', 'num_pm', 'len_nmax', 'len_nmin', 'mautocorrcoef', 'ptpslopes', 'periodicity', 'periodicityr', 'naiveperiod', 'maxvars', 'maxvarsr', 'oeratio', 'amp', 'normamp','mbp', 'mid20', 'mid35', 'mid50', 'mid65', 'mid80', 'percentamp', 'magratio', 'sautocorrcoef', 'autocorrcoef', 'flatmean', 'tflatmean', 'roundmean', 'troundmean', 'roundrat', 'flatrat']
    betterlabels = ['longtermtrend', 'meanmedrat', 'skews', 'varss', 'coeffvar', 'stds', 'numoutliers', 'numnegoutliers', 'numposoutliers', 'numout1s', 'kurt', 'mad', 'maxslope', 'minslope', 'meanpslope', 'meannslope', 'g_asymm', 'rough_g_asymm', 'diff_asymm', 'skewslope', 'varabsslope', 'varslope', 'meanabsslope', 'absmeansecder', 'num_pspikes', 'Negative Spikes (Slope > 3*sigma)', 'num_psdspikes', 'num_nsdspikes','stdratio', 'pstrend', 'Longterm Trendline Crossings', 'Peaks', 'len_nmax', 'len_nmin', 'mautocorrcoef', 'ptpslopes', 'periodicity', 'periodicityr', 'naiveperiod', 'maxvars', 'maxvarsr', 'oeratio', 'amp', 'normamp','mbp', 'mid20', 'mid35', 'mid50', 'mid65', 'mid80', 'percentamp', 'magratio', 'sautocorrcoef', 'autocorrcoef', 'flatmean', 'tflatmean', 'roundmean', 'troundmean', 'roundrat', 'flatrat']
    
    # labeldict connects the variable's code name in listoffeatures to it's presentable name in betterlabels
    labeldict= dict(zip(listoffeatures,betterlabels))
    # mydict connects label to data
    mydict = dict(zip(listoffeatures,ffeatures))
    # labellist keeps track of what each axis should be labelled, initialized with first 3 features
    labellist= [labeldict[listoffeatures[0]],labeldict[listoffeatures[1]],labeldict[listoffeatures[2]]]
    
    # axesdict could probably be accomoplished with a list, I'm partial to the dictionary because it
    # helps me keep things straight.
    axesdict = {'xaxis':ffeatures[0],'yaxis':ffeatures[1],'zaxis':ffeatures[2]}
    
    # Plot initializing stuff
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(axesdict['xaxis'], axesdict['yaxis'], axesdict['zaxis'], c=labels.astype(np.float))
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
    
    rax10,rax11,rax12,rax13,rax14,rax15,rax16,rax17,rax18,rax19,rax110,rax111 = plt.axes(rax1p[0]), plt.axes(rax1p[1]),plt.axes(rax1p[2]),plt.axes(rax1p[3]),plt.axes(rax1p[4]),plt.axes(rax1p[5]),plt.axes(rax1p[6]),plt.axes(rax1p[7]),plt.axes(rax1p[8]),plt.axes(rax1p[9]),plt.axes(rax1p[10]),plt.axes(rax1p[11])
    
    radio10,radio11,radio12,radio13,radio14,radio15,radio16,radio17,radio18,radio19,radio110,radio111 = RadioButtons(rax10, listoffeatures[0:5]),RadioButtons(rax11, listoffeatures[5:10]),RadioButtons(rax12, listoffeatures[10:15]),RadioButtons(rax13, listoffeatures[15:20]),RadioButtons(rax14, listoffeatures[20:25]),RadioButtons(rax15, listoffeatures[25:30]),RadioButtons(rax16, listoffeatures[30:35]),RadioButtons(rax17, listoffeatures[35:40]),RadioButtons(rax18, listoffeatures[40:45]),RadioButtons(rax19, listoffeatures[45:50]),RadioButtons(rax110, listoffeatures[50:55]),RadioButtons(rax111, listoffeatures[55:])
    
    def axis1(label):
        # Clear the figure
        ax.cla()
        # Set the data for the axis to the selected, mydict contains data for label
        axesdict['xaxis'] = [mydict[label]]
        # Plot all that data
        ax.scatter(mydict[label],axesdict['yaxis'],axesdict['zaxis'],c=labels.astype(np.float))
        # Set the axis label
        labellist[0] = label
        ax.set_xlabel(labeldict[labellist[0]])
        ax.set_ylabel(labeldict[labellist[1]])
        ax.set_zlabel(labeldict[labellist[2]])
        # Show it all
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

    rax20 = plt.axes(rax2p[0])
    rax21 = plt.axes(rax2p[1])
    rax22 = plt.axes(rax2p[2])
    rax23 = plt.axes(rax2p[3])
    rax24 = plt.axes(rax2p[4])
    rax25 = plt.axes(rax2p[5])
    rax26 = plt.axes(rax2p[6])
    rax27 = plt.axes(rax2p[7])
    rax28 = plt.axes(rax2p[8])
    rax29 = plt.axes(rax2p[9])
    rax210 = plt.axes(rax2p[10])
    rax211 = plt.axes(rax2p[11])

    radio20 = RadioButtons(rax20, listoffeatures[0:5])
    radio21 = RadioButtons(rax21, listoffeatures[5:10])
    radio22 = RadioButtons(rax22, listoffeatures[10:15])
    radio23 = RadioButtons(rax23, listoffeatures[15:20])
    radio24 = RadioButtons(rax24, listoffeatures[20:25])
    radio25 = RadioButtons(rax25, listoffeatures[25:30])
    radio26 = RadioButtons(rax26, listoffeatures[30:35])
    radio27 = RadioButtons(rax27, listoffeatures[35:40])
    radio28 = RadioButtons(rax28, listoffeatures[40:45])
    radio29 = RadioButtons(rax29, listoffeatures[45:50])
    radio210 = RadioButtons(rax210, listoffeatures[50:55])
    radio211 = RadioButtons(rax211, listoffeatures[55:])
    
    def axis2(label):
        ax.cla()
        axesdict['yaxis'] = [mydict[label]]
        ax.scatter(axesdict['xaxis'],mydict[label],axesdict['zaxis'],c=labels.astype(np.float))
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

    rax30 = plt.axes(rax3p[0])
    rax31 = plt.axes(rax3p[1])
    rax32 = plt.axes(rax3p[2])
    rax33 = plt.axes(rax3p[3])
    rax34 = plt.axes(rax3p[4])
    rax35 = plt.axes(rax3p[5])
    rax36 = plt.axes(rax3p[6])
    rax37 = plt.axes(rax3p[7])
    rax38 = plt.axes(rax3p[8])
    rax39 = plt.axes(rax3p[9])
    rax310 = plt.axes(rax3p[10])
    rax311 = plt.axes(rax3p[11])

    radio30 = RadioButtons(rax30, listoffeatures[0:5])
    radio31 = RadioButtons(rax31, listoffeatures[5:10])
    radio32 = RadioButtons(rax32, listoffeatures[10:15])
    radio33 = RadioButtons(rax33, listoffeatures[15:20])
    radio34 = RadioButtons(rax34, listoffeatures[20:25])
    radio35 = RadioButtons(rax35, listoffeatures[25:30])
    radio36 = RadioButtons(rax36, listoffeatures[30:35])
    radio37 = RadioButtons(rax37, listoffeatures[35:40])
    radio38 = RadioButtons(rax38, listoffeatures[40:45])
    radio39 = RadioButtons(rax39, listoffeatures[45:50])
    radio310 = RadioButtons(rax310, listoffeatures[50:55])
    radio311 = RadioButtons(rax311, listoffeatures[55:])

    def axis3(label):
        ax.cla()
        axesdict['zaxis'] = [mydict[label]]
        ax.scatter(axesdict['xaxis'],axesdict['yaxis'],mydict[label],c=labels.astype(np.float))
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

def feature_calc(filelist):

    files = [line.strip() for line in open(filelist)]

    # Create the appropriate arrays for the features. Length of array determined by number of files.

    numlcs = len(files)
    longtermtrend=np.zeros(numlcs)
    meanmedrat=np.zeros(numlcs)
    skews=np.zeros(numlcs)
    varss=np.zeros(numlcs)
    coeffvar =np.zeros(numlcs)
    stds =np.zeros(numlcs)
    numoutliers =np.zeros(numlcs)
    numnegoutliers =np.zeros(numlcs)
    numposoutliers =np.zeros(numlcs)
    numout1s =np.zeros(numlcs)
    kurt =np.zeros(numlcs)
    mad =np.zeros(numlcs)
    maxslope =np.zeros(numlcs)
    minslope =np.zeros(numlcs)
    meanpslope =np.zeros(numlcs)
    meannslope =np.zeros(numlcs)
    g_asymm=np.zeros(numlcs)
    rough_g_asymm =np.zeros(numlcs)
    diff_asymm =np.zeros(numlcs)
    skewslope =np.zeros(numlcs)
    varabsslope =np.zeros(numlcs)
    varslope =np.zeros(numlcs)
    meanabsslope =np.zeros(numlcs)
    absmeansecder =np.zeros(numlcs)
    num_pspikes=np.zeros(numlcs)
    num_nspikes =np.zeros(numlcs)
    num_psdspikes =np.zeros(numlcs)
    num_nsdspikes=np.zeros(numlcs)
    stdratio =np.zeros(numlcs)
    pstrend =np.zeros(numlcs)
    num_zcross =np.zeros(numlcs)
    num_pm =np.zeros(numlcs)
    len_nmax =np.zeros(numlcs)
    len_nmin =np.zeros(numlcs)
    mautocorrcoef =np.zeros(numlcs)
    ptpslopes =np.zeros(numlcs)
    periodicity =np.zeros(numlcs)
    periodicityr =np.zeros(numlcs)
    naiveperiod =np.zeros(numlcs)
    maxvars =np.zeros(numlcs)
    maxvarsr =np.zeros(numlcs)
    oeratio =np.zeros(numlcs)
    amp = np.zeros(numlcs)
    normamp=np.zeros(numlcs)
    mbp =np.zeros(numlcs)
    mid20 =np.zeros(numlcs)
    mid35 =np.zeros(numlcs)
    mid50 =np.zeros(numlcs)
    mid65 =np.zeros(numlcs)
    mid80 =np.zeros(numlcs)
    percentamp =np.zeros(numlcs)
    magratio=np.zeros(numlcs)
    sautocorrcoef =np.zeros(numlcs)
    autocorrcoef =np.zeros(numlcs)
    flatmean =np.zeros(numlcs)
    tflatmean =np.zeros(numlcs)
    roundmean =np.zeros(numlcs)
    troundmean =np.zeros(numlcs)
    roundrat =np.zeros(numlcs)
    flatrat=np.zeros(numlcs)
    
    # The following runs the program for the list of files in parallel. The number in Pool() should be the number
    # of processors available on the machine's cpu (or 1 less to let the machine keep doing other processes)
    if __name__ == '__main__':    
        p = Pool(6)
        wholecalc = p.map(fcalc,files)
        for i in range(numlcs):
            longtermtrend[i]=wholecalc[i][0]
            meanmedrat[i]=wholecalc[i][1]
            skews[i]=wholecalc[i][2]
            varss[i]=wholecalc[i][3]
            coeffvar[i]=wholecalc[i][4]
            stds[i]=wholecalc[i][5]
            numoutliers[i]=wholecalc[i][6]
            numnegoutliers[i]=wholecalc[i][7]
            numposoutliers[i]=wholecalc[i][8]
            numout1s[i]=wholecalc[i][9] 
            kurt[i]=wholecalc[i][10]
            mad[i]=wholecalc[i][11]
            maxslope[i]=wholecalc[i][12]
            minslope[i]=wholecalc[i][13]
            meanpslope[i]=wholecalc[i][14]
            meannslope[i]=wholecalc[i][15]
            g_asymm[i]=wholecalc[i][16]
            rough_g_asymm[i]=wholecalc[i][17]
            diff_asymm[i]=wholecalc[i][18]
            skewslope[i]=wholecalc[i][19]
            varabsslope[i]=wholecalc[i][20]
            varslope[i]=wholecalc[i][21]
            meanabsslope[i]=wholecalc[i][22]
            absmeansecder[i]=wholecalc[i][23]
            num_pspikes[i]=wholecalc[i][24]
            num_nspikes[i]=wholecalc[i][25]
            num_psdspikes[i]=wholecalc[i][26]
            num_nsdspikes[i]=wholecalc[i][27]
            stdratio[i]=wholecalc[i][28]
            pstrend[i]=wholecalc[i][29]
            num_zcross[i]=wholecalc[i][30]
            num_pm[i]=wholecalc[i][31]
            len_nmax[i]=wholecalc[i][32]
            len_nmin[i]=wholecalc[i][33]
            mautocorrcoef[i]=wholecalc[i][34]
            ptpslopes[i]=wholecalc[i][35]
            periodicity[i]=wholecalc[i][36]
            periodicityr[i]=wholecalc[i][37]
            naiveperiod[i]=wholecalc[i][38]
            maxvars[i]=wholecalc[i][39]
            maxvarsr[i]=wholecalc[i][40]
            oeratio[i]=wholecalc[i][41]
            amp[i]=wholecalc[i][42]
            normamp[i]=wholecalc[i][43]
            mbp[i]=wholecalc[i][44]
            mid20[i]=wholecalc[i][45]
            mid35[i]=wholecalc[i][46]
            mid50[i]=wholecalc[i][47]
            mid65[i]=wholecalc[i][48]
            mid80[i]=wholecalc[i][49]
            percentamp[i]=wholecalc[i][50]
            magratio[i]=wholecalc[i][51]
            sautocorrcoef[i]=wholecalc[i][52]
            autocorrcoef[i]=wholecalc[i][53]
            flatmean[i]=wholecalc[i][54]
            tflatmean[i]=wholecalc[i][55]
            roundmean[i]=wholecalc[i][56]
            troundmean[i]=wholecalc[i][57]
            roundrat[i]=wholecalc[i][58]
            flatrat[i]=wholecalc[i][59]
        p.close()
        p.terminate()
        p.join()
    
    final_features = np.vstack((longtermtrend, meanmedrat, skews, varss, coeffvar, stds, numoutliers, numnegoutliers, numposoutliers, numout1s, kurt, mad, maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes,stdratio, pstrend, num_zcross, num_pm, len_nmax, len_nmin, mautocorrcoef, ptpslopes, periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio, amp, normamp,mbp, mid20, mid35, mid50, mid65, mid80, percentamp, magratio, sautocorrcoef, autocorrcoef, flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat))
    ffeatures = [longtermtrend, meanmedrat, skews, varss, coeffvar, stds, numoutliers, numnegoutliers, numposoutliers, numout1s, kurt, mad, maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes,stdratio, pstrend, num_zcross, num_pm, len_nmax, len_nmin, mautocorrcoef, ptpslopes, periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio, amp, normamp,mbp, mid20, mid35, mid50, mid65, mid80, percentamp, magratio, sautocorrcoef, autocorrcoef, flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat]
    # In order to get all this in the right format for the machine learning we need to set it up as an array where each
    # index is an array of all the features.
    # Data = [ [f1(1),f2(1),f3(1),...,f60(1) ] , [f1(2),f2(2),f3(2),...,f60(2)], ... , [f1(n),...,f60(n)] ]
    data = []
    for i in range(numlcs):
        data.append([])
        for j in range(60):
            data[i].append(final_features[j][i])

    return data,final_features,ffeatures

#final list of features - look up vstack and/or append to consolidate these 

#things that are apparently broken:
#numoutliers   Dan: Seems to be fine, just looks like the sample file has nothing outside 4 sigma?

# print(feature_calc(f))

# data contains the output as arrays of all the features for each lightcurve, good format for kmeans ml
# final_features contains the output as Revant originally had it
# ffeatures contains the output as arrays of all data points for each feature, good format for plotting
data,final_features,ffeatures = feature_calc(f)

#datafile = open('DataLC','w')
#datafile.write(str(data))
#featurefile = open('DataF','w')
#featurefile.write(str(ffeatures))
#kmeans3D(ffeatures,data)
