import numpy as np 
import pandas as pd
np.set_printoptions(threshold='nan')
from scipy import stats
from multiprocessing import Pool,cpu_count
import os
import sys
from datetime import datetime
import pyfits

def fl_files(fl):
    """
    Returns an array with the files in the given filelist
    """
    return [line.strip() for line in open(fl)]

def fl_files_w_path(fl,fitsDir):
    """
    Returns an array with the files with the full path in the given filelist, with the given path
    """
    fcreate = open(fl.replace('.txt',"")+"_completed.txt",'a')
    fcreate.close()
    with open(fl.replace('.txt',"")+"_completed.txt",'r+') as f:
        completed = [fitsDir+'/'+line.strip() for line in f]
    
    return [fitsDir+'/'+line.strip() for line in open(fl) if line not in completed]

def read_kepler_curve(file):
    """
    Given the path of a fits file, this will extract the light curve and normalize it.
    """
    lc = pyfits.getdata(file)
    t = lc.field('TIME')
    f = lc.field('PDCSAP_FLUX')
    err = lc.field('PDCSAP_FLUX_ERR')
    
    err = err[np.isfinite(t)]
    f = f[np.isfinite(t)]
    t = t[np.isfinite(t)]
    err = err[np.isfinite(f)]
    t = t[np.isfinite(f)]
    f = f[np.isfinite(f)]
    err = err/np.median(f)
    nf = f / np.median(f)

    return t, nf, err

def clean_up(fl,fitsDir,in_file='tmp_data.csv'):
    """
    Primarily for a failed run.
    Creates a completed filelist 
    and removes the original filelist 
    if all files have been processed.
    """
    
    files = fl_files(fl)
    df = pd.read_csv(in_file,index_col=0)
    
    with open(fl.replace('.txt',"")+'_completed.txt','a') as f:
        for lc in df.index:
            f.write('%s\n'%lc)
            files.remove(lc)
                
    if files==[]:
        print("All files from original filelist processed, deleting original filelist.")
        os.remove(fl)

    os.remove('tmp_data.csv')
    return

def save_output(out_file,in_file='tmp_data.csv'):
    """
    Reads in the finished data file (tmp_data.csv by default), sorts it, and saves it to the
    specified output csv. Effectively just sorting a csv and renaming it.
    """
    
    df = pd.read_csv(in_file,index_col=0)
    df=df.sort_index()
    df.to_csv(out_file)
    
    return

def featureCalculation(nfile,t,nf,err):
    """
    This is the primary function of this code, it takes in light curve data and returns 60 derived features.
    """
    
    assert len(t)==len(nf) and len(t)==len(err), "t, nf, err arrays must be equal sizes."
        
    try:

        longtermtrend = np.polyfit(t, nf, 1)[0] # Feature 1 (Abbr. F1) overall slope
        yoff = np.polyfit(t, nf, 1)[1] # Not a feature? y-intercept of linear fit
        meanmedrat = np.mean(nf) / np.median(nf) # F2
        skews = stats.skew(nf) # F3
        varss = np.var(nf) # F4
        coeffvar = np.std(nf)/np.mean(nf) #F5
        stds = np.std(nf) #F6

        corrnf = nf - longtermtrend*t - yoff #this removes any linear slope to lc so you can look at just troughs - is this a sign err tho?
        # D: I don't think there's a sign error

        posthreshold = np.mean(nf)+4*np.std(nf)
        negthreshold = np.mean(nf)-4*np.std(nf)

        numposoutliers,numnegoutliers,numout1s=0,0,0
        for j in range(len(nf)):
            # First checks if nf[j] is outside of 1 sigma
            if abs(np.mean(nf)-nf[j])>np.std(nf):
                numout1s += 1 #F7
                if nf[j]>posthreshold:
                    numposoutliers += 1 #F8
                elif nf[j]<negthreshold:
                    numnegoutliers += 1 #F9
        numoutliers=numposoutliers+numnegoutliers #F10
        
        kurt = stats.kurtosis(nf)

        mad = np.median([abs(nf[j]-np.median(nf)) for j in range(len(nf))])

        # slopes array contains features 13-30
        
        # delta nf/delta t
        slopes=[(nf[j+1]-nf[j])/(t[j+1]-t[j]) for j in range(len(nf)-1)]

        #corrslopes removes the longterm linear trend (if any) and then looks at the slope
        corrslopes=[(corrnf[j+1]-corrnf[j])/(t[j+1]-t[j]) for j in range (len(corrnf)-1)] #F11
        meanslope = np.mean(slopes) #F12

        # by looking at where the 99th percentile is instead of just the largest number,
        # I think it avoids the extremes which might not be relevant (might be unreliable data)
        # Is the miniumum slope the most negative one, or the flattest one? Answer: Most negative
        maxslope=np.percentile(slopes,99) #F13
        minslope=np.percentile(slopes,1) #F14

        # Separating positive slopes and negative slopes
        # Should both include the 0 slope? I'd guess there wouldn't be a ton, but still...
        pslope=[slope for slope in slopes if slope>=0]
        nslope=[slope for slope in slopes if slope<=0]
        # Looking at the average (mean) positive and negative slopes
        if len(pslope)==0:
            meanpslope=0
        else:
            meanpslope=np.mean(pslope) #F15

        if len(nslope)==0:
            meannslope=0
        else:
            meannslope=np.mean(nslope) #F16

        # Quantifying the difference in shape.
        if meannslope==0:
            g_asymm = 10
        else:
            g_asymm=meanpslope / meannslope #F17

        # Won't this be skewed by the fact that both pslope and nslope have all the 0's? Eh
        if len(nslope)==0:
            rough_g_asymm=10
        else:
            rough_g_asymm=len(pslope) / len(nslope) #F18

        # meannslope is inherently negative, so this is the difference btw the 2
        diff_asymm=meanpslope + meannslope #F19
        skewslope = stats.skew(slopes) #F20
        absslopes=[abs(slope) for slope in slopes]
        meanabsslope=np.mean(absslopes) #F21
        varabsslope=np.var(absslopes) #F22
        varslope=np.var(slopes) #F23

        #secder = Second Derivative
        # Reminder for self: the slope is "located" halfway between the flux and time points, 
        # so the delta t in the denominator is accounting for that.
        # secder = delta slopes/delta t, delta t = ((t_j-t_(j-1))+(t_(j+1)-t_j))/2
        #secder=[(slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2) for j in range(1, len(slopes)-1)]
        # after algebraic simplification:
        secder=[2*(slopes[j]-slopes[j-1])/(t[j+1]-t[j-1]) for j in range(1, len(slopes)-1)]
        
        #abssecder=[abs((slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2)) for j in range (1, len(slopes)-1)]
        # simplification:

        abssecder=np.abs(np.array(secder))
        absmeansecder=np.mean(abssecder) #F24
        if len(pslope)==0:
            pslopestds=0
        else:
            pslopestds=np.std(pslope)
        if len(nslope)==0:
            nslopesstds=0
            stdratio=10 # arbitrary ratio chosen, the ratio will normally be ~1, so 10 seems big enough.
        else:
            nslopestds=np.std(nslope)
            stdratio=pslopestds/nslopestds

        sdstds=np.std(secder)
        meanstds=np.mean(secder)


        num_pspikes,num_nspikes,num_psdspikes,num_nsdspikes=0,0,0,0

        for slope in slopes:
            if slope>=meanpslope+3*pslopestds:
                num_pspikes+=1 #F25
            elif slope<=meanslope-3*nslopestds:
                num_nspikes+=1 #F26

        for sder in secder:
            if sder>=4*sdstds:
                num_psdspikes+=1 #F27
            elif sder<=-4*sdstds:
                num_nsdspikes+=1 #F28
        if nslopestds==0:
            stdratio=10
        else:
            stdratio = pslopestds / nslopestds #F29

        # The ratio of postive slopes with a following postive slope to the total number of points.
        pstrendcount = 0
        for j,slope in enumerate(slopes[:-1]):
            if slope > 0 and slopes[j+1]>0:
                pstrendcount += 1

        pstrend=pstrendcount/len(slopes) #F30

        # Checks if the flux crosses the zero line.
        zcrossind= [j for j in range(len(nf)-1) if corrnf[j]*corrnf[j+1]<0]
        num_zcross = len(zcrossind) #F31

        plusminus=[j for j in range(1,len(slopes)) if (slopes[j]<0)&(slopes[j-1]>0)]
        num_pm = len(plusminus)

        # This looks up the local maximums. Adds a peak if it's the largest within 10 points on either side.
        # Q: Is there a way to do this and take into account drastically different periodicity scales?

        naivemax,nmax_times = [],[]
        naivemins = []
        for j in range(len(nf)):
            if nf[j] == max(nf[max(j-10,0):min(j+10,len(nf)-1)]):
                naivemax.append(nf[j])
                nmax_times.append(t[j])
            elif nf[j] == min(nf[max(j-10,0):min(j+10,len(nf)-1)]):
                naivemins.append(nf[j])
        len_nmax=len(naivemax) #F33
        len_nmin=len(naivemins) #F34    
        if len(naivemax)>2:
            mautocorrcoef = np.corrcoef(naivemax[:-1], naivemax[1:])[0][1] #F35
        else:
            mautocorrcoef = 0
        """peak to peak slopes"""
        ppslopes = [abs((naivemax[j+1]-naivemax[j])/(nmax_times[j+1]-nmax_times[j])) \
                    for j in range(len(naivemax)-1)]
        if len(ppslopes)==0:
            ptpslopes = 0
        else:
            ptpslopes=np.mean(ppslopes) #F36

        maxdiff=[nmax_times[j+1]-nmax_times[j] for j in range(len(naivemax)-1)]

        if len(maxdiff)==0:
            periodicity=0
            periodicityr=0
            naiveperiod=0
        else:
            periodicity=np.std(maxdiff)/np.mean(maxdiff) #F37
            periodicityr=np.sum(abs(maxdiff-np.mean(maxdiff)))/np.mean(maxdiff) #F38
            naiveperiod=np.mean(maxdiff) #F39
        if len(naivemax)==0:
            maxvars=0
            maxvarsr=0
        else:
            maxvars = np.std(naivemax)/np.mean(naivemax) #F40
            maxvarsr = np.sum(abs(naivemax-np.mean(naivemax)))/np.mean(naivemax) #F41

        emin = naivemins[::2] # even indice minimums
        omin = naivemins[1::2] # odd indice minimums
        meanemin = np.mean(emin)
        if len(omin)==0:
            meanomin=0
        else:
            meanomin = np.mean(omin)
        oeratio = meanomin/meanemin #F42

        # amp here is actually amp_2 in revantese
        # 2x the amplitude (peak-to-peak really), the 1st percentile will be negative, so it's really adding magnitudes
        amp = np.percentile(nf,99)-np.percentile(nf,1) #F43
        normamp = amp / np.mean(nf) #this should prob go, since flux is norm'd #F44

        # ratio of points within 10% of middle to total number of points 
        mbp = len([nf[j] for j in range(len(nf))\
                   if (nf[j] < (np.median(nf) + 0.1*amp)) \
                   & (nf[j] > (np.median(nf)-0.1*amp))]) / len(nf) #F45

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

        autocorrcoef = np.corrcoef(nf[:-1], nf[1:])[0][1] #F54

        sautocorrcoef = np.corrcoef(slopes[:-1], slopes[1:])[0][1] #F55

        #measures the slope before and after the maximums
        flatness = [np.mean(slopes[max(0,j-6):min(max(0,j-1), len(slopes)-1):1])\
                    - np.mean(slopes[max(0,j):min(j+4, len(slopes)-1):1])\
                    for j in range(6,len(slopes)-6) \
                    if nf[j] in naivemax]

        if len(flatness)==0: flatmean=0
        else: flatmean = np.nansum(flatness)/len(flatness) #F55

        #measures the slope before and after the minimums
        # trying flatness w slopes and nf rather than "corr" vals, despite orig def in RN's program
        tflatness = [-np.mean(slopes[max(0,j-6):min(max(j-1,0),len(slopes)-1):1])\
                     + np.mean(slopes[j:min(j+4,len(slopes)-1):1])\
                     for j in range(6,len(slopes)-6)\
                     if nf[j] in naivemins] 

        # tflatness for mins, flatness for maxes
        if len(tflatness)==0: 
            tflatmean=0
        else: 
            tflatmean = np.nansum(tflatness) / len(tflatness) #F56

        roundness = [np.mean(secder[max(0,j-6):j:1])\
                      +np.mean(secder[j:min(j+6,len(secder)-1):1])\
                      for j in range(6,len(secder)-6)\
                      if nf[j] in naivemax]

        if len(roundness)==0: 
            roundmean=0
        else: 
            roundmean = np.nansum(roundness) / len(roundness) #F57

        troundness = [np.mean(secder[max(0,j-6):j])\
                      +np.mean(secder[j:min(j+6,len(secder)-1)])\
                      for j in range(6,len(secder)-6)\
                      if nf[j] in naivemins]

        if len(troundness)==0:
            troundmean=0
        else:
            troundmean = np.nansum(troundness)/len(troundness) #F58

        if troundmean==0 and roundmean==0: 
            roundrat=1
        elif troundmean==0: 
            roundrat=10
        else: 
            roundrat = roundmean / troundmean #F59

        if flatmean==0 and tflatmean==0: 
            flatrat=1
        elif tflatmean==0: 
            flatrat=10
        else: 
            flatrat = flatmean / tflatmean #F60

        ndata = np.array([longtermtrend, meanmedrat, skews, varss, coeffvar, stds, \
                 numoutliers, numnegoutliers, numposoutliers, numout1s, kurt, mad, \
                 maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, \
                 diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, \
                 num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes,stdratio, pstrend, \
                 num_zcross, num_pm, len_nmax, len_nmin, mautocorrcoef, ptpslopes, \
                 periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio, \
                 amp, normamp, mbp, mid20, mid35, mid50, \
                 mid65, mid80, percentamp, magratio, sautocorrcoef, autocorrcoef, \
                 flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat])
        if __name__=="__main__":
            fts = ["longtermtrend", "meanmedrat", "skews", "varss", "coeffvar", "stds", \
                     "numoutliers", "numnegoutliers", "numposoutliers", "numout1s", "kurt", "mad", \
                     "maxslope", "minslope", "meanpslope", "meannslope", "g_asymm", "rough_g_asymm", \
                     "diff_asymm", "skewslope", "varabsslope", "varslope", "meanabsslope", "absmeansecder", \
                     "num_pspikes", "num_nspikes", "num_psdspikes", "num_nsdspikes","stdratio", "pstrend", \
                     "num_zcross", "num_pm", "len_nmax", "len_nmin", "mautocorrcoef", "ptpslopes", \
                     "periodicity", "periodicityr", "naiveperiod", "maxvars", "maxvarsr", "oeratio", \
                     "amp", "normamp", "mbp", "mid20", "mid35", "mid50", \
                     "mid65", "mid80", "percentamp", "magratio", "sautocorrcoef", "autocorrcoef", \
                     "flatmean", "tflatmean", "roundmean", "troundmean", "roundrat", "flatrat"]

            df = pd.DataFrame([ndata],index=[nfile.replace(fitsDir+'/',"")],columns=fts)

            with open('tmp_data.csv','a') as f:
                df.to_csv(f,header=False)
            return
        else:
            return nfile,ndata
        
    except TypeError:
        kml_log = 'kml_log'
        os.system('echo %s ... TYPE ERROR >> %s'%(nfile.replace(fitsDir+'/',""),kml_log))
        return 

def features_from_fits(nfile):
    t,nf,err = read_kepler_curve(nfile)
    # t = time
    # err = error
    # nf = normalized flux.
    features = featureCalculation(nfile,t,nf,err)
    if __name__=="__main__":
        return
    else:
        return features
    
def features_from_filelist(fl,fitsDir,of,numCpus = cpu_count(), verbose=False):
    """
    This method calculates the features of the given filelist from the fits files located in fitsDir.
    All output is saved to a temporary csv file called tmp_data.csv.
    Run save_output(output file) and clean_up(filelist, fits file directory) to save to the desired location,
    and to clean up the filelist and tmp_data.csv. 
    
    Returns pandas dataframe of output.
    """
    
    df = pd.DataFrame({
        "longtermtrend":[], "meanmedrat":[], "skews":[], "varss":[], "coeffvar":[], "stds":[], \
        "numoutliers":[], "numnegoutliers":[], "numposoutliers":[], "numout1s":[], "kurt":[], "mad":[], \
        "maxslope":[], "minslope":[], "meanpslope":[], "meannslope":[], "g_asymm":[], "rough_g_asymm":[], \
        "diff_asymm":[], "skewslope":[], "varabsslope":[], "varslope":[], "meanabsslope":[], "absmeansecder":[], \
        "num_pspikes":[], "num_nspikes":[], "num_psdspikes":[], "num_nsdspikes":[],"stdratio":[], "pstrend":[], \
        "num_zcross":[], "num_pm":[], "len_nmax":[], "len_nmin":[], "mautocorrcoef":[], "ptpslopes":[], \
        "periodicity":[], "periodicityr":[], "naiveperiod":[], "maxvars":[], "maxvarsr":[], "oeratio":[], \
        "amp":[], "normamp":[], "mbp":[], "mid20":[], "mid35":[], "mid50":[], \
        "mid65":[], "mid80":[], "percentamp":[], "magratio":[], "sautocorrcoef":[], "autocorrcoef":[], \
        "flatmean":[], "tflatmean":[], "roundmean":[], "troundmean":[], "roundrat":[], "flatrat":[]})

    with open('tmp_data.csv','w') as f:df.to_csv(f)
    # files with path.
    files = fl_files_w_path(fl,fitsDir)
    if verbose:
        print("Using %s cpus to calculate features..."%numCpus)
    p = Pool(numCpus)
    # Method saves to tmp_data.csv file to save on system memory
    p.map(features_from_fits,files)
    p.close()
    p.join()
    if verbose:
        print("Features have been calculated")
        print("Saving output to %s"%of)
    save_output(of)
    if verbose:print("Output saved, cleaning up...")
    clean_up(fl,fitsDir)
    if verbose:print("Done.")
    
    if __name__=="__main__":
        return
    else:
        return pd.read_csv(of)
    
if __name__=="__main__":
    """
    If this is run as a script, the following will parse the arguments it is fed, 
    or prompt the user for input.
    
    python keplerml.py path/to/filelist path/to/fits_file_directory path/to/output_file
    """
    # fl - filelist, a txt file with file names, 1 per line
    if sys.argv[1]:
        fl = sys.argv[1]
    else:
        fl = raw_input("Input path: ")
    print("Reading %s..."%fl)

    if sys.argv[2]:
        fitsDir = sys.argv[2]
    else:
        fitsDir = raw_input("Fits files directory path: ")
    # of - output file
    if sys.argv[3]:
        of = sys.argv[3]
    else:
        of = raw_input('Output path: ')
        if of  == "":
            print("No output path specified, saving to output.csv in local folder.")
            of = 'output.csv'
    
    features_from_filelist(fl,fitsDir,of,verbose=True)
    