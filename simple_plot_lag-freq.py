#!/usr/bin/env
"""
Plots the lag-frequency spectrum.

Example call:
python simple_plot_lag-freq.py ./cygx1_lag-freq.fits -o "./cygx1" --ext "png"

Enter   python simple_plot_lag-freq.py -h   at the command line for help.
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from astropy.table import Table
import argparse
import subprocess
import numpy as np

__author__ = "Abigail Stevens <A.L.Stevens at uva.nl>"
__year__ = "2016"

################################################################################
def main(lag_file, out_base="./out", plot_ext="eps"):

    lag_table = Table.read(lag_file)

    # x_err = np.repeat(lag_table.meta['DF'], len(lag_table['FREQUENCY']))

    font_prop = font_manager.FontProperties(size=20)

    plot_file = out_base + "_lag-freq." + plot_ext
    print("Lag-frequency spectrum: %s" % plot_file)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), dpi=300, tight_layout=True)

    ax.plot([lag_table['FREQUENCY'][0], lag_table['FREQUENCY'][-1]], [0, 0],
            lw=1.5, ls='dashed', c='black')
    # 	ax.plot([freq[0], freq[-1]],[np.pi,np.pi], lw=1.5, ls='dashed', c='black')
    # 	ax.plot([freq[0], freq[-1]],[-np.pi,-np.pi], lw=1.5, ls='dashed', c='black')

    # ax.errorbar(lag_table['FREQUENCY'], lag_table['PHASE_LAG'], xerr=x_err,
    #             yerr=lag_table['PHASE_LAG_ERR'], marker='o', ms=10, mew=2,
    #             mec='blue', fillstyle='none', ecolor='blue', elinewidth=2,
    #             capsize=0)

    ax.errorbar(lag_table['FREQUENCY'], lag_table['TIME_LAG'],
                yerr=lag_table['TIME_LAG_ERR'], marker='o', ms=10, mew=2,
                mec='blue', fillstyle='none', ecolor='blue', elinewidth=2,
                capsize=0)

    ax.set_xlabel('Frequency (Hz)', fontproperties=font_prop)
    ax.set_ylabel('Time lag (s)', fontproperties=font_prop)
    # 	ax.set_ylabel('Phase lag (radians)', fontproperties=font_prop)

    # ax.set_xlim(lo_freq, up_freq)
    ax.set_xlim(3, 7)
    ax.set_ylim(-1, 1)
    # ax.set_ylim(1.3 * np.min(lag_table['TIME_LAG']),
    #             1.3 * np.max(lag_table['TIME_LAG']))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(which='major', width=1.5, length=7)
    ax.tick_params(which='minor', width=1.5, length=4)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    title = "Lag-frequency spectrum"
    ax.set_title(title, fontproperties=font_prop)

    plt.savefig(plot_file)
    # plt.show()
    plt.close()
    # subprocess.call(['open', plot_file])


################################################################################
if __name__ == "__main__":
    #########################################
    ## Parse input arguments and call 'main'
    #########################################

    parser = argparse.ArgumentParser(usage="python simple_cross_spectra.py "
            "lag_file [OPTIONAL ARGUMENTS]",
            description=__doc__,
            epilog="For optional arguments, default values are given in "\
            "brackets at end of description.")

    parser.add_argument('lag_file', help="The FITS file with the lag-frequency "
                                         "spectrum saved as an astropy table.")

    parser.add_argument('-o', '--out', default="./out", dest='outbase',
                        help="The base name for plot file. Extension will be "
                             "appended. [./out]")
    parser.add_argument('--ext', default="eps", dest='plot_ext',
                        help="The plot extension. Do not include the dot. "
                             "[eps]")
    args = parser.parse_args()

    main(args.lag_file, args.outbase, args.plot_ext)
