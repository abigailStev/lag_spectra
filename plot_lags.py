## DEPRECIATED. Now incorporated into get_lags.py.

import argparse
import numpy as np
from scipy import fftpack
from datetime import datetime
import subprocess
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import tools

__author__ = "Abigail Stevens"
__year__ = "2014-2015"
__description__ = "DEPRECIATED. Now incorporated into get_lags.py."

"""
        plot_lags.py

DEPRECIATED. Now incorporated into get_lags.py.

Written in Python 2.7.

"""


def plot_lag-frequency():


def plot_lag-energy():
    pass




################################################################################
if __name__ == "__main__":
	
	###########################
	## Parsing input arguments
	###########################

	parser = argparse.ArgumentParser(usage="python plot_lags.py tab_file \
[-o out_root]", description="DEPRECIATED. Now incorporated into get_lags.py.", \
epilog="For optional arguments, default values \
are given in brackets at end of description.")
		
	parser.add_argument('tab_file', help="The table with lag-frequency and lag-\
energy data, in .fits format.")
	
	parser.add_argument('-o', '--outroot', dest='out_root', default="./test", \
help="The output file name for the power spectrum plot. [./test]")
	
	parser.add_argument('-p', '--prefix', required=False, dest='prefix', \
default="--", help="The identifying prefix for the file (proposal ID or object \
nickname). [--]")

	args = parser.parse_args()
	
	assert args.tab_file[-4:].lower() == "fits", \
		"ERROR: Input file must have .fits format."
	
	font_prop = font_manager.FontProperties(size=16)
	
	######################################
	## Reading in the cross spectrum data
	######################################
	
	try:
		fits_hdu = fits.open(args.tab_file)
	except IOError:
		print "\tERROR: File does not exist: %s" % args.tab_file
		exit()
		
	detchans = int(fits_hdu[0].header['DETCHANS'])	
	low_freq = float(fits_hdu[0].header['LOWFREQ'])
	hi_freq = float(fits_hdu[0].header['HIGHFREQ'])
# 	dead_chan = int(fits_hdu[0].header['DEADCHAN'])
	f_data = fits_hdu[1].data
	e_data = fits_hdu[2].data
	fits_hdu.close()


	###########################################################
	## Lag-frequency spectrum (for individual energy channels)
	###########################################################
	
	## Loop through energy channels (for event mode, 0 - 63 incl)
	for i in xrange(15,16):
	
		plot_file = args.out_root + "_lag-freq_" + str(i) + ".png"
		print "Lag-frequency spectrum: %s" % plot_file
		channel_mask = f_data.field('CHANNEL') == i  ## Make data mask for the energy channel(s) i want
		f_data_i = f_data[channel_mask]
		freq = f_data_i.field('FREQUENCY')
		phase = f_data_i.field('PHASE')
		phase_err = f_data_i.field('PHASE_ERR')
		time_lag = f_data_i.field('TIME_LAG')
		time_lag_err = f_data_i.field('TIME_LAG_ERR')

		fig, ax = plt.subplots(1,1)
		ax.plot([freq[0], freq[-1]],[0,0], lw=1.5, ls='dashed', c='black')
# 		ax.plot([freq[0], freq[-1]],[np.pi,np.pi], lw=1.5, ls='dashed', c='black')
# 		ax.plot([freq[0], freq[-1]],[-np.pi,-np.pi], lw=1.5, ls='dashed', c='black')
# 		ax.plot(freq, phase, lw=3, c='blue', ls='steps-mid')
# 		ax.errorbar(freq, phase, yerr=phase_err, lw=3, c='blue', \
# 			ls='steps-mid', elinewidth=2, capsize=2)
		ax.errorbar(freq, time_lag, yerr=time_lag_err, lw=3, c='blue', \
			ls='steps-mid', capsize=2, elinewidth=2)
		ax.set_xlabel('Frequency (Hz)', fontproperties=font_prop)
		ax.set_ylabel('Time lag (s)', fontproperties=font_prop)
# 		ax.set_ylabel('Phase lag (radians)', fontproperties=font_prop)
		ax.set_xlim(low_freq, hi_freq)
		ax.set_ylim(-0.3, 0.3)
# 		ax.set_ylim(-6, 6)
		ax.tick_params(axis='x', labelsize=14)
		ax.tick_params(axis='y', labelsize=14)
		ax.set_title("Lag-frequency spectrum, %s, channel %s/%s" % (args.prefix, \
			str(i), str(detchans)), fontproperties=font_prop)

		plt.savefig(plot_file, dpi=120)
# 		plt.show()
		plt.close()
	## End of for-loop through energy channels
	
	#####################################################
	## Lag-energy spectrum (for a given frequency range)
	#####################################################
	
	plot_file = args.out_root + "_lag-energy.png"
	print "Lag-energy spectrum: %s" % plot_file

	phase = e_data.field('PHASE')
	phase_err = e_data.field('PHASE_ERR')
	time_lag = e_data.field('TIME_LAG')
	time_lag_err = e_data.field('TIME_LAG_ERR')
	chans = e_data.field('CHANNEL')
	
	## Deleting the values at energy channel 10	
	phase = np.delete(phase, 10)
	phase_err = np.delete(phase_err, 10)
	time_lag = np.delete(time_lag, 10)
	time_lag_err = np.delete(time_lag_err, 10)
	chans = np.delete(chans, 10)
	
	fig, ax = plt.subplots(1,1)
	ax.plot([0,detchans],[0,0], lw=1.5, ls='dashed', c='black')
# 	ax.plot([0,detchans],[np.pi,np.pi], lw=1.5, ls='dashed', c='black')
# 	ax.plot([0,detchans],[-np.pi,-np.pi], lw=1.5, ls='dashed', c='black')
# 	ax.plot(range(0,detchans), phase, lw=3, c='red', ls='steps-mid')
# 	ax.errorbar(range(0,detchans), phase, yerr=phase_err, lw=3, c='red', \
# 		ls="steps-mid", elinewidth=2, capsize=2)
	ax.errorbar(chans, time_lag, yerr=time_lag_err, ls='none', c='red', \
		marker='x', markersize=10, elinewidth=2, capsize=2)
	ax.set_xlabel('Energy channel (0-%s)' % str(detchans-1), fontproperties=font_prop)
	ax.set_ylabel('Time lag (s)', fontproperties=font_prop)
# 	ax.set_ylabel('Phase lag (radians)', fontproperties=font_prop)
	ax.set_xlim(2,25)
# 	ax.set_ylim(-0.03, 0.03)
# 	ax.set_ylim(-np.pi, np.pi)
# 	ax.set_ylim(-6, 6)
	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	ax.set_title("Lag-energy spectrum, %s, %s - %s Hz" % (args.prefix, \
		str(low_freq), str(hi_freq)), fontproperties=font_prop)

	plt.savefig(plot_file, dpi=150)
# 	plt.show()
	plt.close()
	
################################################################################
