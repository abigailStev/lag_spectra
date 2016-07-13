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
from astropy.table import Table
import os.path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
import matplotlib.colors as colors
import itertools

__author__ = 'Abigail Stevens <A.L.Stevens at uva.nl>'
__year__ = '2015-2016'

HOME_DIR = os.path.expanduser("~")
DAY = datetime.now().strftime("%y%m%d")

PLOT_EXT = "eps"
LO_FREQ = 4.0
UP_FREQ = 7.0
LO_ENERGY = 3.0
UP_ENERGY = 20.0

DETCHANS = 64

################################################################################
def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    From https://docs.python.org/2/library/itertools.html#recipes
    Used when reading lines in the file so I can peek at the next line.

    Parameters
    ----------
    an iterable, like a list or an open file

    Returns
    -------
    The next two items in an iterable, like in the example a few lines above.

    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

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
        Dir + base name of the plot file. "_lag-energy.[PLOT_EXT]" are appended
        to it.

    Returns
    -------
    nothing

    """

    colours = [colors.cnames['darkblue'],
               colors.cnames['darkviolet'],
               colors.cnames['coral'],
               "black"]

    # colours = [colors.cnames['darkred'],
    #            colors.cnames['darkcyan'],
    #            colors.cnames['deeppink'],
    #            "black"]

    markers = ['^',
               'x',
               's',
               'o']

    #######################
    ## Setting up the plot
    #######################

    font_prop = font_manager.FontProperties(size=20)
    energies = np.loadtxt(HOME_DIR + "/Reduced_data/" + prefix +
                          "/energies.txt")
    energy_list = [np.mean([x, y]) for x,y in pairwise(energies)]
    energy_err = [np.abs(a-b) for (a,b) in zip(energy_list, energies[0:-1])]

    plot_file = plot_root + "_lag-energy." + PLOT_EXT
    print "Lag-energy spectrum: %s" % plot_file

    e_chans = np.arange(0, DETCHANS)

    ###################
    ## Making the plot
    ###################

    fig, ax = plt.subplots(1, 1, figsize=(10,7.5), dpi=300, tight_layout=True)

    ax.hlines(0.0, 3, 21, linestyle='dashed', lw=2, color='black')

    ## Deleting the values at energy channel 10 for RXTE PCA event-mode data
    if DETCHANS == 64:
        e_chans = np.delete(e_chans, 10)
        energy_list = np.delete(energy_list, 10)
        energy_err = np.delete(energy_err, 10)
    i = 0

    for in_file in in_file_list:

        # try:
        #     fits_hdu = fits.open(in_file)
        # except IOError:
        #     print "\tERROR: File does not exist: %s" % in_file
        #     exit()
        # tlag = fits_hdu[2].data.field('TIME_LAG')
        # tlag_err = fits_hdu[2].data.field('TIME_LAG_ERR')

        try:
            lag_table = Table.read(in_file, format='fits', hdu=2)  # HDU 2 for energy lags
        except IOError:
            print "\tERROR: File does not exist: %s" % in_file
            exit()
        tlag = lag_table['TIME_LAG']
        tlag_err = lag_table['TIME_ERR']

        ## Deleting the values at energy channel 10 for RXTE PCA event-mode data
        if DETCHANS == 64:
            tlag = np.delete(tlag, 10)
            tlag_err = np.delete(tlag_err, 10)

            if labels[i].lower() == "data":
                ax.errorbar(energy_list[2:26], tlag[2:26],
                            xerr=energy_err[2:26], yerr=tlag_err[2:26],
                            ls='none', marker='o', ms=10, mew=2, mec='black',
                            mfc='black', ecolor='black', elinewidth=3,
                            capsize=0, label=labels[i])

            else:
                if markers[i] == 'x':
                    ax.errorbar(energy_list[2:26], tlag[2:26],
                                xerr=energy_err[2:26], yerr=tlag_err[2:26],
                                lw=3, drawstyle='steps-mid', ls='none', ms=11,
                                marker=markers[i], mec=colours[i], mew=2,
                                color=colours[i], fillstyle='none',
                                ecolor=colours[i], elinewidth=3, capsize=0,
                                label=labels[i])
                else:
                    ax.errorbar(energy_list[2:26], tlag[2:26],
                                xerr=energy_err[2:26], yerr=tlag_err[2:26],
                                lw=3, drawstyle='steps-mid', ls='none', ms=8,
                                mew=2, marker=markers[i], mec=colours[i],
                                color=colours[i], fillstyle='none',
                                ecolor=colours[i], elinewidth=3, capsize=0,
                                label=labels[i])

        else:
            ax.errorbar(energy_list, tlag, xerr=energy_err, yerr=tlag_err,
                        ls='none', marker=markers[i], ms=10, mew=2,
                        mec=colours[i], fillstyle='none', ecolor=colours[i],
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
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(which='major', width=1.5, length=7)
    ax.tick_params(which='minor', width=1.5, length=4)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    # ax.set_title("Lag-energy spectrum", fontproperties=font_prop)

    ## The following legend code was found on stack overflow I think
    # legend = ax.legend(loc='upper left')
    # for label in legend.get_texts():
    #     label.set_fontsize(18)
    # for label in legend.get_lines():
    #     label.set_linewidth(2)  # the legend line width

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=20,
            borderpad=0.5, labelspacing=0.5, borderaxespad=0.5)
    # ax.text(18, -0.008, 'b', fontsize=36)
    plt.savefig(plot_file)
    # 	plt.show()
    plt.close()
    subprocess.call(['open', plot_file])
    # subprocess.call(['cp', plot_file,
    #         "/Users/abigailstevens/Dropbox/Academic/Conferences_and_talks/HEAD_Florida2016/"])


################################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage="python overplot_lag-energy.py "\
            "prefix", description=__doc__)

    parser.add_argument('prefix', help="Identifying prefix of data (object "\
            "nickname or data ID).")
    args = parser.parse_args()
    prefix = args.prefix

    lag_dir = HOME_DIR + "/Dropbox/Research/lag_spectra/out_lags/" + prefix
    sim_dir = HOME_DIR + "/Dropbox/Research/simulate/out_sim/" + prefix

    # plot_root = "/Users/abigailstevens/Dropbox/Academic/Conferences_and_talks/HEAD_NaplesFL2016/GX339_151204_t64_64sec_overpl"

    # in_file_list = [HOME_DIR + "/Dropbox/Research/lag_spectra/out_lags/GX339-BQPO/GX339-BQPO_151204_t64_64sec_adj_lag.fits",
    #                 HOME_DIR + "/Dropbox/Research/lag_spectra/out_lags/GX339-4HzCQPO/GX339-4HzCQPO_160318_t64_64sec_lag.fits"]
    # labels = ["Type B", "Type C"]

    # plot_root = lag_dir + "/" + prefix + "_151204_t64_64sec_overpl"
    #
    # in_file_list = [sim_dir + "/bootstrapped/" + prefix + \
    #                     "_151204_1BB-FS-G-Tin-fzs-fzNbb_lag.fits",
    # 				sim_dir + "/bootstrapped/" + prefix + \
    #                     "_151204_2BB-FS-G-kT-fzs-fzNbb8857-2_lag.fits",
    # 				sim_dir + "/bootstrapped/" + prefix + \
    #                     "_151204_pBB-FS-G-p-fzs-fzNbb_lag.fits",
    #                 lag_dir + "/" + prefix + "_151204_t64_64sec_adj_lag.fits"]
    #
    # labels = ["Model 1", "Model 2", "Model 3", "Data"]

    # plot_root = lag_dir + "/" + prefix + "_151204_t64_64sec_overpl_bad"

    # in_file_list = [sim_dir + "/" + prefix + "_151204_1BB-FS-G_lag.fits",
    # 				sim_dir + "/" + prefix + "_151204_1BB-FS-G-E-fzs_lag.fits",
    # 				sim_dir + "/" + prefix + "_151204_1BB-FS-G-NE-fzs_lag.fits",
    #                 lag_dir + "/" + prefix + "_151204_t64_64sec_adj_lag.fits"]

    # labels = [r"$\Gamma$, $f_{scatt}$",
    #           r"$\Gamma$, $f_{scatt}$, $E_{line}$",
    #           r"$\Gamma$, $f_{scatt}$, $N_{line}$",
    #           "Data"]

    plot_root = lag_dir + "/" + prefix + "_160712_t64_64sec_adj_overpl"

    in_file_list = [lag_dir + "/" + prefix + "_160711_t64_64sec-3-10Hz_wh_adj_lag.fits",
                    lag_dir + "/" + prefix + "_160711_t64_64sec-3-7Hz_adj_lag.fits",
                    lag_dir + "/" + prefix + "_160712_t64_64sec-7-10Hz_adj_lag.fits"]

    labels = ["3-10 Hz", "3-7 Hz", "7-10 Hz"]


    ################################################
    ## Calls method above to actually make the plot
    ################################################

    make_plot(in_file_list, labels, plot_root)


