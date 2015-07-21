#!/usr/bin/env

import argparse
import subprocess
import numpy as np
from astropy.io import fits
import os.path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter

import tools
import get_lags as lags

__author__ = 'Abigail Stevens, A.L.Stevens at uva.nl'

in_file = "/Users/abigailstevens/Dropbox/Research/simulate/out_sim/FAKE-GX339B_150721_nopoiss_cs.fits"
out_file = "/Users/abigailstevens/Dropbox/Research/simulate/out_sim/FAKE-GX339B_150721_nopoiss_lag.fits"
plot_root = "/Users/abigailstevens/Dropbox/Research/simulate/out_sim/FAKE-GX339B_150721_nopoiss"
prefix = "FAKE-GX339B"
plot_ext = "eps"
lo_freq = 4.0
up_freq = 7.0
lo_energy = 3.0
up_energy = 20.0

## Get necessary information and data from the input file
freq, cs_avg, power_ci, power_ref, dt, n_bins, detchans, num_seconds, \
        num_seg, mean_rate_ci, mean_rate_ref, evt_list = lags.get_inputs(in_file)

######################
## Computing the lags
######################

f_phase, f_err_phase, f_tlag, f_err_tlag, e_phase,e_err_phase, e_tlag, \
        e_err_tlag = lags.compute_lags(freq, cs_avg, power_ci, power_ref, dt, \
        n_bins, detchans, num_seconds, num_seg, mean_rate_ci, mean_rate_ref, \
        lo_freq, up_freq, lo_energy, up_energy)

##########
## Output
##########

lags.fits_out(out_file, in_file, evt_list, dt, n_bins, num_seg, detchans,
         lo_freq, up_freq, lo_energy, up_energy, mean_rate_ci,
         mean_rate_ref, freq, f_phase, f_err_phase, f_tlag, f_err_tlag,
         e_phase, e_err_phase, e_tlag, e_err_tlag)

############
## Plotting
############

lags.plot_lag_energy(plot_root, plot_ext, prefix, e_phase, e_err_phase, e_tlag,
                e_err_tlag, lo_freq, up_freq, detchans)