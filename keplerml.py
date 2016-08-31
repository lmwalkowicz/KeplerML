# L.M. Walkowicz
# Rewrite of Revant's feature calculations, plus additional functions for vetting outliers
from datetime import datetime
startTime = datetime.now()

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
from multiprocessing import Pool,cpu_count
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons
import sys
if sys.version_info[0] < 3:
    from Tkinter import Tk
else:
    from tkinter import Tk

from tkFileDialog import askopenfilename,askdirectory

print('Select the filelist')
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filelist = askopenfilename() # show an "Open" dialog box and return the path to the selected file

print('Select the fits files location (must all be in one directory)')
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
fitsDir = askdirectory() # show an "Open" dialog box and return the path to the selected file

identifier=str(raw_input('Choose a unique identifier: '))

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
    
    f = f[np.isfinite(t)]
    t = t[np.isfinite(t)]
    t = t[np.isfinite(f)]
    f = f[np.isfinite(f)]
    
    nf = f / np.median(f)

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

def feature_calc(filelist):
    
    files = [fitsDir+'/'+line.strip() for line in open(filelist)]
    
    # The following runs the program for the list of files in parallel. The number in Pool() should be the number
    # of processors available on the machine's cpu (or 1 less to let the machine keep doing other processes)
    if __name__ == '__main__':
        numcpus = cpu_count()
        if numcpus > 1:
            # Leave 1 cpu open for system tasks.
            usecpus = numcpus-1
        else:
            usecpus = 1
            
        p = Pool(usecpus)
        data = p.map(fcalc,files)
        p.close()
        p.terminate()
        p.join()

    return data

#things that are apparently broken:
#numoutliers   Dan: Seems to be fine, just looks like the sample file has nothing outside 4 sigma?

# 'data' contains the output as arrays of all the features for each lightcurve, necessary for clustering
data = feature_calc(filelist)

# This will save the calculated features as numpy arrays in a .npy file, which can be imported via np.load(file)
np.save(identifier+'dataByLightCurve',data)

print datetime.now()-startTime