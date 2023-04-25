"""Mapping between variables & names."""

###################################################################################################
###################################################################################################

## ELECTRODE LOCATIONS

LOC_MAP = {'LA' : 'AMY',
           'RA' : 'AMY',
           'LAH' : 'AH',
           'RAH' : 'AH',
           'LPH' : 'PH',
           'RPH' : 'PH'}

SIDE_MAP = {'LA' : 'left',
            'RA' : 'right',
            'LAH' : 'left',
            'RAH' : 'right',
            'LPH' : 'left',
            'RPH' : 'right'}

## ANALYSES

ANALYSIS_MAP = {
    
    # 1-back Task
    'nback_stimulus' : {'stat' : 'baseline_tvalue_abs',
                        'label' : '1B Stimulus t-value',
                        'sig' : 'is_baseline'},
    
    'nback_id' : {'stat' : 'id_fvalue',
                  'label' : '1B Identity F-value',
                  'sig' : 'is_id'},
    
    # TH task
    'th_stimulus' : {'stat' : 'fr_t_val_full_abs',
                    'label' : 'TH Stimulus t-value',
                    'sig' : 'is_chest'},
    
    'th_serial' : {'stat' : 'serial_anova',
                   'label' : 'TH Serial F-value',
                   'sig' : 'is_serial'},
    
    'th_target' : {'stat' : 'target_anova',
                   'label' : 'TH Target F-value',
                   'sig' : 'is_target'},
}