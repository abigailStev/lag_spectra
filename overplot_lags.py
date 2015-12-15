#!/usr/bin/env
"""
Plots multiple lag-energy spectra (from different observations, simulations,
etc.) on the same plot.

Change lag_dir and sim_dir to suit your machine and setup.

Example call: python overplot_lag-energy.py cygx1
"""
import argparse
import subprocess
import numpy as np
from astropy.io import fits
import os.path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from datetime import datetime

import tools  ## in https://github.com/abigailStev/whizzy_scripts
# import get_lags as lags

__author__ = 'Abigail Stevens <A.L.Stevens at uva.nl>'
__year__ = '2015'

HOME_DIR = os.path.expanduser("~")
DAY = datetime.now().strftime("%y%m%d")

PLOT_EXT = "eps"
LO_FREQ = 4.0
UP_FREQ = 7.0
LO_ENERGY = 3.0
UP_ENERGY = 20.0

DETCHANS = 64


################################################################################
def make_plot(in_file_list, labels, plot_root, prefix="GX339-BQPO"):
    """
    Plots multiple lag plots in one figure.

    Parameters
    ----------
    in_file_list : list of str
        1-D list of the lag files to plot.

    labels : list of str
        1-D list of the line labels for each lag. Must be same length as
        in_file_list.

    plot_root : str
        Dir + base name of the plot file. _lag-energy.[PLOT_EXT] are appended to
        it.

    Returns
    -------
    nothing

    """

    colours = ["black",
               "blue",
               "green",
               "magenta",
               "orange"]

    markers = ['o',
               'd',
               '*',
               '.',
               'x']

    #######################
    ## Setting up the plot
    #######################

    font_prop = font_manager.FontProperties(size=20)
    energies = np.loadtxt(HOME_DIR + "/Reduced_data/" + prefix +
                          "/energies.txt")
    energy_list = [np.mean([x, y]) for x,y in tools.pairwise(energies)]
    energy_err = [np.abs(a-b) for (a,b) in zip(energy_list, energies[0:-1])]

    plot_file = plot_root + "_lag-energy." + PLOT_EXT
    print "Lag-energy spectrum: %s" % plot_file

    e_chans = np.arange(0, DETCHANS)


    ###################
    ## Making the plot
    ###################

    fig, ax = plt.subplots(1, 1, figsize=(10,7.5), dpi=300, tight_layout=True)

    ax.hlines(0.0, 3, 21, linestyle='dashed', lw=2, color='gray')

    ## Deleting the values at energy channel 10 for RXTE PCA event-mode data
    if DETCHANS == 64:
        e_chans = np.delete(e_chans, 10)
        energy_list = np.delete(energy_list, 10)
        energy_err = np.delete(energy_err, 10)
    i = 0

    for in_file in in_file_list:
        try:
            fits_hdu = fits.open(in_file)
        except IOError:
            print "\tERROR: File does not exist: %s" % in_file
            exit()

        # print fits_hdu.info()

        tlag = fits_hdu[2].data.field('TIME_LAG')

        tlag_err = fits_hdu[2].data.field('TIME_LAG_ERR')

        ## Deleting the values at energy channel 10 for RXTE PCA event-mode data
        if DETCHANS == 64:
            tlag = np.delete(tlag, 10)
            tlag_err = np.delete(tlag_err, 10)

            ax.errorbar(energy_list[2:26], tlag[2:26], xerr=energy_err[2:26],
                    yerr=tlag_err[2:26], ls='none', marker=markers[i], ms=10,
                    mew=2, mec=colours[i], mfc=colours[i], ecolor=colours[i],
                    elinewidth=2, capsize=0, label=labels[i])
        else:
            ax.errorbar(energy_list, tlag, xerr=energy_err,
                    yerr=tlag_err, ls='none', marker=markers[i], ms=10, mew=2,
                    mec=colours[i], mfc=colours[i], ecolor=colours[i],
                    elinewidth=2, capsize=0, label=labels[i])
        i += 1

    # ax.plot([0,DETCHANS],[0,0], lw=1.5, ls='dashed', c='black')
    # 	ax.plot([0,DETCHANS],[np.pi,np.pi], lw=1.5, ls='dashed', c='black')
    # 	ax.plot([0,DETCHANS],[-np.pi,-np.pi], lw=1.5, ls='dashed', c='black')
    # 	ax.errorbar(e_chans, phase, yerr=err_phase, lw=3, c='red', \
    # 		ls="steps-mid", elinewidth=2, capsize=2)

    ax.set_xlabel('Energy (keV)', fontproperties=font_prop)
    ax.set_xlim(3, 21)
    # ax.set_xlim(0.3, 10)
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

    plt.savefig(plot_file)
    # 	plt.show()
    plt.close()
    subprocess.call(['open', plot_file])
    subprocess.call(['cp', plot_file, \
            "/Users/abigailstevens/Dropbox/Research/CCF_paper1/"])


################################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage="python overplot_lag-energy.py "\
            "prefix", description=__doc__)

    parser.add_argument('prefix', help="Identifying prefix of data (object "\
            "nickname or data ID).")
    args = parser.parse_args()
    prefix = args.prefix

    # ## Get necessary information and data from the input file
    # freq, cs_avg, power_ci, power_ref, dt, n_bins, DETCHANS, num_seconds, \
    #         num_seg, mean_rate_ci, mean_rate_ref, evt_list = lags.get_inputs(in_file)
    #
    # #########################################
    # ## Computing the lags of the fundamental
    # #########################################
    #
    # f_phase_1, f_err_phase_1, f_tlag_1, f_err_tlag_1, e_phase_1, e_err_phase_1, \
    #         e_tlag_1, e_err_tlag_1 = lags.compute_lags(freq, cs_avg, power_ci, \
    #         power_ref, dt, n_bins, DETCHANS, num_seconds, num_seg, mean_rate_ci, \
    #         mean_rate_ref, lo_freq, up_freq, lo_energy, up_energy)
    #
    # #########################################
    # ## Writing the output
    # #########################################
    #
    # # lags.fits_out(out_file, in_file, evt_list, dt, n_bins, num_seg, DETCHANS,
    # #          lo_freq, up_freq, lo_energy, up_energy, mean_rate_ci,
    # #          mean_rate_ref, freq, f_phase_1, f_err_phase_1, f_tlag_1, f_err_tlag_1,
    # #          e_phase_1, e_err_phase_1, e_tlag_1, e_err_tlag_1)
    #
    #
    # ###########################################
    # ## Computing the lags of only the harmonic
    # ###########################################
    #
    #
    # in_file = sim_dir + "/FAKE-" + prefix + "_150722_nopoiss_cs.fits"
    #
    # ## Get necessary information and data from the input file
    # freq, cs_avg, power_ci, power_ref, dt, n_bins, DETCHANS, num_seconds, \
    #         num_seg, mean_rate_ci, mean_rate_ref, evt_list = lags.get_inputs(in_file)
    #
    # f_phase_2, f_err_phase_2, f_tlag_2, f_err_tlag_2, e_phase_2, e_err_phase_2, \
    #         e_tlag_2, e_err_tlag_2 = lags.compute_lags(freq, cs_avg, power_ci, \
    #         power_ref, dt, n_bins, DETCHANS, num_seconds, num_seg, mean_rate_ci, \
    #         mean_rate_ref, lo_freq, up_freq, lo_energy, up_energy)
    #
    # ###########################################################
    # ## Computing the lags of both the fundamental and harmonic
    # ###########################################################
    #
    # ## Get necessary information and data from the input file
    # freq, cs_avg, power_ci, power_ref, dt, n_bins, DETCHANS, num_seconds, \
    #         num_seg, mean_rate_ci, mean_rate_ref, evt_list = lags.get_inputs(in_file)
    #
    # f_phase_3, f_err_phase_3, f_tlag_3, f_err_tlag_3, e_phase_3, e_err_phase_3, \
    #         e_tlag_3, e_err_tlag_3 = lags.compute_lags(freq, cs_avg, power_ci, \
    #         power_ref, dt, n_bins, DETCHANS, num_seconds, num_seg, mean_rate_ci, \
    #         mean_rate_ref, lo_freq, up_freq, lo_energy, up_energy)

    lag_dir = HOME_DIR + "/Dropbox/Research/lag_spectra/out_lags/" + prefix
#     sim_dir = HOME_DIR + "/Dropbox/Research/simulate/out_sim/" + prefix
    plot_root = lag_dir + "/" + prefix + "_151119_t64_64sec_overpl_lag"

    in_file_list = [lag_dir + "/" + prefix + "_151124_t64_64sec_adj_lag.fits",
    				lag_dir + "/" + prefix + "_150814_t64_64sec_adj_lag.fits"]
    				
    # labels = [r"PL norm & index, $\chi^{2}=165.8$",
    #           r"+ iron line energy, $\chi^{2}=144.4$",
    #           # r"+ iron line norm, $\chi^{2}=78.0$",
    #           r"+ BB temp, $\chi^{2}=37.8$",
    #           "Data"]
    labels = ["New", "Old"]

    ################################################
    ## Calls method above to actually make the plot
    ################################################

    make_plot(in_file_list, labels, plot_root)


