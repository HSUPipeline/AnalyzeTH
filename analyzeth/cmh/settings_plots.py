
# Seaborn style
seaborn_style = 'darkgrid'

# Fonts
f_small = 12
f_mid = 14
f_large = 16

# General plot parameters
plot_params = {
        'xtick.major.size'   : 2,
        'xtick.major.width'  : 1,
        'ytick.major.size'   : 2,
        'ytick.major.width'  : 1,
        'xtick.bottom'       : True,
        'ytick.left'         : True,
        'xtick.direction'    : 'in',
        'ytick.direction'    : 'in',
        'font.size'          : f_small,
        'axes.titlesize'     : f_large,
        'axes.labelsize'     : f_mid,
        'legend.fontsize'    : f_small,
        'figure.titlesize'   : f_large,
        'xtick.labelsize'    : f_small,
        'ytick.labelsize'    : f_small,
        'xtick.major.pad'    : 10,
        'ytick.major.pad'    : 10,
        'axes.labelpad'      : 15,
        'axes.titlepad'      : 15
}

# Settings dict
plot_settings = {
    'seaborn_style'     : seaborn_style,
    'fonts'             : {
        'f_small'       : f_small,
        'f_mid'         : f_mid,
        'f_large'       : f_large
    },
    'plot_params'       : plot_params
}