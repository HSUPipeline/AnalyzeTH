import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analyzeth.cmh.headDirection.headDirectionPlots import plot_bootstrap_replicates_PDF_hist 

def hd_anova():
    return



# Bootstrap equations
def draw_bs_replicates(data,func,size):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)
    
    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        bs_sample = np.random.choice(data,size=len(data))
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample)
    
    return bs_replicates



def bootstrap_ci_from_surrogates(surrogates, func = np.mean, num_bootstraps = 10000, ci=95, verbose = False, plot_verbose = False):
    """
    Bootstrap for 95 % confidence intervals from surrogate data
    
    Parameters
    ----------
    surrogates: 2d arr
        array containing surrogates, 
            x = hz or firing count reading for each degreen bin (of arbitrary size - i.e. can be 1 or 
            90 degrees etc.), default 360 bins (one per degree)
            y = each surrogate, defualt 1000 surrogates
    
    func: function 
        estimator to use for bootstrapping, default np.mean, can do np.std etc

    num_bootstraps: int
        number of bootstraps to run, default 10,000
    
    ci: int
        confidence interval to calculate, default 95%
        will give upper and lower bounds, i.e. the 2.5% and 97,5% bounds for calculated metrics (default = np.mean)
    
    Returns
    -------
    boostrap_confidence_intervals: 2d arr
        2d array containing the (95%) confidence interval determined for each bin
    """


    # Setup 
    num_bins = surrogates.shape[1]
    ##df = pd.DataFrame(surrogates).melt()
    ##df.rename(columns={'variable' : 'bin', 'value': 'hz'}, inplace = True)

    # Bootstrap 
    bootstrap_ci95s = np.zeros([num_bins, 2]) 
    # print(f'bs array shape: \t {bootstrap_ci95s.shape}')

    for ix in range(num_bins):    
        #surrogates_bin = df[df['bin']==ix]
        surrogates_bin = surrogates[:,ix]
        bootstrap_replicates_bin = draw_bs_replicates(surrogates_bin, func, num_bootstraps)

        # Confidence interval
        ci_lower = (100 - ci)/2
        ci_upper = 100 - ci_lower
        conf_interval = np.percentile(bootstrap_replicates_bin,[ci_lower,ci_upper])
        bootstrap_ci95s[ix,:] = conf_interval

        # verbose
        if verbose and ix%100 ==0:
            print(f'Computing boostrap for bin {ix}')
            if plot_verbose:
                plot_bootstrap_replicates_PDF_hist(bootstrap_replicates_bin)

    print('in bs func -- ci95s shape', bootstrap_ci95s.shape)
    return bootstrap_ci95s 

def compute_significant_bins(hd_histogram, confidence_interval):
    """
    Check if sample from each bin is above ci95. If so
    record bin as significant only if the 5 bins on either side
    are also above the upper ci95 threshold

    Parameters
    ----------
    hd_histogram: 1d arr
        array of bin values (if normalized to occupancy then Hz) for each head direction

    confidence_interval: 2d arr
        array of lower [:,0] and upper [:,1] confidence bounds for each bin based on surrogates
        default is to use bootsrapping with np.mean for determining 95% confidence interval
    
    Returns
    -------
    significant_bins: 1d arr
        boolean array indicating which bins are found to be significant (above ci upper bound for
        bin + 5 bins on each side)

    """
    hd_mask = hd_histogram > confidence_interval[:,1]
    hd_mask_circle = np.hstack([hd_mask, hd_mask[:10]])

    num_bins = len(hd_histogram)
    significant_bins = np.zeros(num_bins)
    for ix in range(num_bins):
        if hd_mask_circle[ix] == True:
            sig = True
            for jx in range(ix-5,ix+6):
                if hd_mask_circle[jx] == False:
                    sig = False
            significant_bins[ix] = sig
    return significant_bins

