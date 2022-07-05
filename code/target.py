""""Functions for spatial target analyses.

ToDo:
- Fix up interim updates
- Check different functions and merge common stuff
"""

from functools import partial
from collections import Counter

import numpy as np

from spiketools.spatial.occupancy import compute_nbins
from spiketools.stats.anova import create_dataframe, fit_anova
from spiketools.utils.extract import get_range, get_value_by_time, get_value_by_time_range

# import local code
from analysis import get_spike_positions

###################################################################################################
###################################################################################################

# Define ANOVA model
MODEL = 'fr ~ C(target_bin)'
FEATURE = 'C(target_bin)'
COLUMNS = ['target_bin', 'fr']

# Create functions for target model
create_df_target = partial(create_dataframe, columns=COLUMNS)
fit_anova_target = partial(fit_anova, formula=MODEL, feature=FEATURE)

###################################################################################################
###################################################################################################

def get_trial_target(nav_starts, openings, spikes, chest_bins, 
                     chest_trials, chest_xbin, chest_ybin):
  """Get the binned target firing, per trial."""
    
    n_trials = len(openings)
    n_bins = compute_nbins(chest_bins)

    # Collect firing per chest location for each trial
    target_bins_all = np.zeros([n_trials, n_bins])
    for t_ind in range(n_trials):

        # Get chest and opening events of current trial
        t_openings = openings[t_ind]
        t_mask = chest_trials == t_ind

        # Get navigation start & end and restrict spikes to this range
        t_st = nav_starts[t_ind]
        t_en = t_openings[-1]
        t_spikes = get_range(spikes, t_st, t_en)

        # Compute firing rate per target bin per trial
        seg_times = np.diff(np.insert(t_openings, 0, t_st))
        count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
        count.update(np.digitize(t_spikes, t_openings))

        frs = np.array(list(count.values())) / seg_times
        
        cur_ch_xbin = chest_xbin[t_mask]
        cur_ch_ybin = chest_ybin[t_mask]
        
        target_bins = np.zeros(chest_bins)
        for fr, xbin, ybin in zip(frs, cur_ch_xbin, cur_ch_ybin):
            target_bins[xbin, ybin] = fr

        target_bins_all[t_ind, :] = target_bins.flatten()
        
    return target_bins_all
  
  
def compute_spatial_target_bins(chest_occupancy, target_bins_all, chest_bins, set_nan=False):
    """Compute the binned firing rate based on spatial target."""
    
    chest_occupancy[chest_occupancy == 0.] = np.nan
    chest_occ = chest_occupancy.flatten()
    
    # Sum up the firing rate per bin of all trials 
    target_bins_sum = np.sum(target_bins_all,axis=0)
    target_bins_sum[target_bins_sum == 0.] = np.nan
    
    # Compute the averaged firing rate per bin across trials
    target_bins = ((target_bins_sum / chest_occ).reshape(chest_bins)).transpose()
    target_bins = np.nan_to_num(target_bins)
    
    if set_nan:
        target_bins[target_bins == 0.] = np.nan
    
    return target_bins
