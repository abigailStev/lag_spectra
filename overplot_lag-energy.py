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

in_file = "/Users/abigailstevens/Dropbox/Research/simulate/out_sim/FAKE-GX339B_150722_nopoiss_cs.fits"
out_file = "/Users/abigailstevens/Dropbox/Research/simulate/out_sim/FAKE-GX339B_150722_nopoiss_lag.fits"
plot_root = "/Users/abigailstevens/Dropbox/Research/simulate/out_sim/FAKE-GX339B_150722_nopoiss"
prefix = "FAKE-GX339B"

# in_file = "/Users/abigailstevens/Dropbox/Research/lags/out_lags/GX339-BQPO_150720_t64_64sec_adj_cs.fits"
# out_file = "/Users/abigailstevens/Dropbox/Research/lags/out_lags/GX339-BQPO_150720_t64_64sec_adj_lag.fits"
# plot_root = "/Users/abigailstevens/Dropbox/Research/lags/out_lags/GX339-BQPO_150720_t64_64sec_adj"
# prefix = "GX339-BQPO"

plot_ext = "eps"
lo_freq = 4.0
up_freq = 7.0
lo_energy = 3.0
up_energy = 20.0

in_file = "/Users/abigailstevens/Dropbox/Research/simulate/out_sim/FAKE-GX339B_150721_nopoiss_cs.fits"

## Get necessary information and data from the input file
freq, cs_avg, power_ci, power_ref, dt, n_bins, detchans, num_seconds, \
        num_seg, mean_rate_ci, mean_rate_ref, evt_list = lags.get_inputs(in_file)

#########################################
## Computing the lags of the fundamental
#########################################

f_phase_1, f_err_phase_1, f_tlag_1, f_err_tlag_1, e_phase_1, e_err_phase_1, \
        e_tlag_1, e_err_tlag_1 = lags.compute_lags(freq, cs_avg, power_ci, \
        power_ref, dt, n_bins, detchans, num_seconds, num_seg, mean_rate_ci, \
        mean_rate_ref, lo_freq, up_freq, lo_energy, up_energy)

#########################################
## Writing the output
#########################################

# lags.fits_out(out_file, in_file, evt_list, dt, n_bins, num_seg, detchans,
#          lo_freq, up_freq, lo_energy, up_energy, mean_rate_ci,
#          mean_rate_ref, freq, f_phase_1, f_err_phase_1, f_tlag_1, f_err_tlag_1,
#          e_phase_1, e_err_phase_1, e_tlag_1, e_err_tlag_1)


###########################################
## Computing the lags of only the harmonic
###########################################


in_file = "/Users/abigailstevens/Dropbox/Research/simulate/out_sim/FAKE-GX339B_150722_nopoiss_cs.fits"

## Get necessary information and data from the input file
freq, cs_avg, power_ci, power_ref, dt, n_bins, detchans, num_seconds, \
        num_seg, mean_rate_ci, mean_rate_ref, evt_list = lags.get_inputs(in_file)

f_phase_2, f_err_phase_2, f_tlag_2, f_err_tlag_2, e_phase_2, e_err_phase_2, \
        e_tlag_2, e_err_tlag_2 = lags.compute_lags(freq, cs_avg, power_ci, \
        power_ref, dt, n_bins, detchans, num_seconds, num_seg, mean_rate_ci, \
        mean_rate_ref, lo_freq, up_freq, lo_energy, up_energy)

###########################################################
## Computing the lags of both the fundamental and harmonic
###########################################################
in_file = "/Users/abigailstevens/Dropbox/Research/lags/out_lags/GX339-BQPO_150727_t64_64sec_adj_cs.fits"

## Get necessary information and data from the input file
freq, cs_avg, power_ci, power_ref, dt, n_bins, detchans, num_seconds, \
        num_seg, mean_rate_ci, mean_rate_ref, evt_list = lags.get_inputs(in_file)

f_phase_3, f_err_phase_3, f_tlag_3, f_err_tlag_3, e_phase_3, e_err_phase_3, \
        e_tlag_3, e_err_tlag_3 = lags.compute_lags(freq, cs_avg, power_ci, \
        power_ref, dt, n_bins, detchans, num_seconds, num_seg, mean_rate_ci, \
        mean_rate_ref, lo_freq, up_freq, lo_energy, up_energy)



############
## Plotting
############

font_prop = font_manager.FontProperties(size=18)
energies = np.loadtxt(\
        "/Users/abigailstevens/Reduced_data/GX339-BQPO/energies.txt")
energy_list = [np.mean([x, y]) for x,y in tools.pairwise(energies)]
energy_err = [np.abs(a-b) for (a,b) in zip(energy_list, energies[0:-1])]

plot_file = plot_root + "_lag-energy." + plot_ext
print "Lag-energy spectrum: %s" % plot_file

e_chans = np.arange(0, detchans)

## Deleting the values at energy channel 10
e_phase_1 = np.delete(e_phase_1, 10)
e_err_phase_1 = np.delete(e_err_phase_1, 10)
e_tlag_1 = np.delete(e_tlag_1, 10)
e_err_tlag_1 = np.delete(e_err_tlag_1, 10)
e_phase_2 = np.delete(e_phase_2, 10)
e_err_phase_2 = np.delete(e_err_phase_2, 10)
e_tlag_2 = np.delete(e_tlag_2, 10)
e_err_tlag_2 = np.delete(e_err_tlag_2, 10)
e_phase_3 = np.delete(e_phase_3, 10)
e_err_phase_3= np.delete(e_err_phase_3, 10)
e_tlag_3 = np.delete(e_tlag_3, 10)
e_err_tlag_3 = np.delete(e_err_tlag_3, 10)
e_chans = np.delete(e_chans, 10)
energy_list = np.delete(energy_list, 10)
energy_err = np.delete(energy_err, 10)

fig, ax = plt.subplots(1, 1, figsize=(10,7.5), dpi=300)
# ax.plot([0,detchans],[0,0], lw=1.5, ls='dashed', c='black')
# 	ax.plot([0,detchans],[np.pi,np.pi], lw=1.5, ls='dashed', c='black')
# 	ax.plot([0,detchans],[-np.pi,-np.pi], lw=1.5, ls='dashed', c='black')
# 	ax.errorbar(e_chans, phase, yerr=err_phase, lw=3, c='red', \
# 		ls="steps-mid", elinewidth=2, capsize=2)

ax.hlines(0.0, 3, 21, linestyle='dashed', lw=2, color='gray')
# # ax.errorbar(energy_list, e_tlag_1, yerr=e_err_tlag_1, ls='none', marker='x', \
# #             ms=8, mew=1.5, mec='black', ecolor='black', \
# #             elinewidth=1.5, capsize=1.5)
# ax.plot(energy_list, e_phase_1, ls='none', marker='x', \
#             ms=8, mew=1.5, mec='red', label=r"Simulation, tied BB kT")
# ax.plot(energy_list, e_phase_2, ls='none', marker='*', \
#             ms=8, mew=1.5, mec='blue', label=r"Simulation, varying BB kT")
# ax.errorbar(energy_list, e_phase_3, yerr=e_err_phase_3, ls='none', marker='+', \
#             ms=8, mew=1.5, mec='black', ecolor='black', elinewidth=1.5, \
#             capsize=1.5, label="Data")
ax.errorbar(energy_list[2:26], e_tlag_1[2:26], xerr=energy_err[2:26], ls='none',
            marker='x', ms=10, mew=2, mec='red', mfc='red', ecolor='red',
            elinewidth=2, capsize=0, label=r"Simulation, tied BB kT")
ax.errorbar(energy_list[2:26], e_tlag_2[2:26], xerr=energy_err[2:26], ls='none',
            marker='*', ms=11, mew=2, mec='blue', mfc='blue', ecolor='blue',
            elinewidth=2, capsize=0, label=r"Simulation, varying BB kT")
ax.errorbar(energy_list[2:26], e_tlag_3[2:26], xerr=energy_err[2:26],
            yerr=e_err_tlag_3[2:26], ls='none', marker='o', ms=5, mew=2,
            mec='black', mfc='black', ecolor='black', elinewidth=2, capsize=0,
            label="Data")

ax.set_xlabel('Energy (keV)', fontproperties=font_prop)
ax.set_xlim(3, 21)
# ax.set_ylim(-0.009, 0.016)
ax.set_xscale('log')
x_maj_loc = [5, 10, 20]
# y_maj_loc = [-0.005, 0, 0.005, 0.01, 0.015]
ax.set_xticks(x_maj_loc)
# ax.set_yticks(y_maj_loc)
xLocator = MultipleLocator(1)  ## loc of minor ticks on x-axis
yLocator = MultipleLocator(0.001)  ## loc of minor ticks on y-axis
ax.xaxis.set_minor_locator(xLocator)
ax.yaxis.set_minor_locator(yLocator)
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.set_ylabel('Time lag (s)', fontproperties=font_prop)
# ax.set_ylabel('Phase lag (radians)', fontproperties=font_prop)
ax.set_ylim(-0.01, 0.017)
# ax.set_ylim(-0.4, 0.5)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
# ax.set_title("Lag-energy spectrum", fontproperties=font_prop)

## The following legend code was found on stack overflow I think
legend = ax.legend(loc='upper left')
for label in legend.get_texts():
	label.set_fontsize(18)
for label in legend.get_lines():
	label.set_linewidth(2)  # the legend line width

fig.set_tight_layout(True)
plt.savefig(plot_file)
# 	plt.show()
plt.close()
subprocess.call(['open', plot_file])
subprocess.call(['cp', plot_file, "/Users/abigailstevens/Dropbox/Research/CCF_paper1/"])