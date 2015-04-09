import argparse
import subprocess
import numpy as np
from astropy.io import fits
from scipy import fftpack
from datetime import datetime
import os.path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import tools

__author__ = "Abigail Stevens"
__author_email__ = "A.L.Stevens at uva.nl"
__year__ = "2015"
__description__ = "Computes the phase lag and time lag of energy channels of \
interest with a reference energy band from the average cross spectrum. Can \
average over frequency and over energy."

"""
		get_lags.py

Enter   python get_lags.py -h   at the command line for help.

Written in Python 2.7.

All scientific modules imported above, as well as python 2.7, can be downloaded
in the Anaconda package, https://store.continuum.io/cshop/anaconda/

"""

################################################################################
def get_inputs(in_file):
	"""
			get_inputs
	
	Gets cross spectrum, CI power spectrum, ref power spectrum, and necessary
	constants from the input file.
	
	"""
	try:
		fits_hdu = fits.open(in_file)
	except IOError:
		print "\tERROR: File does not exist: %s" % in_file
		exit()
		
	evt_list = fits_hdu[0].header['EVTLIST']
	dt = float(fits_hdu[0].header['DT'])
	n_bins = int(fits_hdu[0].header['N_BINS'])
	num_seg = int(fits_hdu[0].header['SEGMENTS'])
	exposure = float(fits_hdu[0].header['EXPOSURE'])
	detchans = int(fits_hdu[0].header['DETCHANS']) # == number of interest bands
	rate_ci = np.asarray(fits_hdu[0].header['RATE_CI'])
	rate_ref = float(fits_hdu[0].header['RATE_REF'])
	num_seconds = n_bins * dt
	
	cs_data = fits_hdu[1].data
	powci_data = fits_hdu[2].data
	powref_data = fits_hdu[3].data
	
	try:
		cs_avg = np.reshape(cs_data.field('CROSS'), (n_bins/2+1, detchans), \
			order='C')
		power_ci = np.reshape(powci_data.field('POWER'), (n_bins/2+1, detchans),\
			order='C')
	except ValueError:
		cs_avg = np.reshape(cs_data.field('CROSS'), (n_bins, detchans), \
			order='C')
		power_ci = np.reshape(powci_data.field('POWER'), (n_bins, detchans), \
			order='C')
		
	power_ref = powref_data.field('POWER')
	freq = powref_data.field('FREQUENCY')
	
	return freq, cs_avg, power_ci, power_ref, dt, n_bins, detchans, \
		num_seconds, num_seg, rate_ci, rate_ref, evt_list
## End of function 'get_inputs'	


################################################################################
def fits_out(out_file, in_file, evt_list, dt, n_bins, num_seg, detchans, \
	lo_freq, up_freq, lo_energy, up_energy, mean_rate_ci, mean_rate_ref, freq, \
	phase, err_phase, tlag, err_tlag, e_phase, e_err_phase, e_tlag, e_err_tlag): 
	"""
			fits_out
	"""
	
	chan = np.arange(0, detchans)
	f_bins = np.repeat(freq, len(chan))  
	
	print "Output sent to: %s" % out_file
	
	## Making FITS header (extension 0)
	prihdr = fits.Header()
	prihdr.set('TYPE', "Lag-frequency and lag-energy spectral data")
	prihdr.set('DATE', str(datetime.now()), "YYYY-MM-DD localtime")
	prihdr.set('EVTLIST', evt_list)
	prihdr.set('CS_DATA', in_file)
	prihdr.set('DT', dt, "seconds")
	prihdr.set('N_BINS', n_bins, "time bins per segment")
	prihdr.set('SEGMENTS', num_seg, "segments in the whole light curve")
	prihdr.set('EXPOSURE', num_seg * n_bins * dt,
		"seconds, of light curve")
	prihdr.set('DETCHANS', detchans, "Number of detector energy channels")
	prihdr.set('LAG_LF', lo_freq, "Hz")
	prihdr.set('LAG_UF', up_freq, "Hz")
	prihdr.set('LAG_LE', lo_energy, "Detector channels")
	prihdr.set('LAG_UE', up_energy, "Detector channels")
	prihdr.set('RATE_CI', str(mean_rate_ci.tolist()), "counts/second")
	prihdr.set('RATE_REF', mean_rate_ref, "counts/second")
	prihdu = fits.PrimaryHDU(header=prihdr)

	## Making FITS table for lag-frequency plot (extension 1)
	col1 = fits.Column(name='FREQUENCY', format='D', array=f_bins)
	col2 = fits.Column(name='PHASE', unit='radians', format='D',
		array=phase.flatten('C'))
	col3 = fits.Column(name='PHASE_ERR', unit='radians', format='D',
		array=err_phase.flatten('C'))
	col4 = fits.Column(name='TIME_LAG', unit='s', format='D',
		array=tlag.flatten('C'))
	col5 = fits.Column(name='TIME_LAG_ERR', unit='s', format='D',
		array=err_tlag.flatten('C'))
	cols = fits.ColDefs([col1, col2, col3, col4, col5])
	tbhdu1 = fits.BinTableHDU.from_columns(cols)
	
	## Making FITS table for lag-energy plot (extension 2)
	col1 = fits.Column(name='PHASE', unit='radians', format='D', array=e_phase)
	col2 = fits.Column(name='PHASE_ERR', unit='radians', format='D', \
		array=e_err_phase)
	col3 = fits.Column(name='TIME_LAG', unit='s', format='D', array=e_tlag)
	col4 = fits.Column(name='TIME_LAG_ERR', unit='s', format='D', \
		array=e_err_tlag)
	col5 = fits.Column(name='CHANNEL', unit='', format='I', \
		array=chan)
	cols = fits.ColDefs([col1, col2, col3, col4, col5])
	tbhdu2 = fits.BinTableHDU.from_columns(cols)
	
	## If the file already exists, remove it
	assert out_file[-4:].lower() == "fits", \
		'ERROR: Output file must have extension ".fits".'
	if os.path.isfile(out_file):
		subprocess.call(["rm", out_file])
	
	## Writing to a FITS file
	thdulist = fits.HDUList([prihdu, tbhdu1, tbhdu2])
	thdulist.writeto(out_file)	
	
	return 
## End of function 'fits_out'


################################################################################
def get_phase_err(cs_avg, power_ci, power_ref, n, M):
	"""
			get_phase_err
	
	Computes the error on the complex phase (in radians). Power should not be 
	noise-subtracted.
	
	"""
	with np.errstate(all='ignore'):
		a = power_ci * power_ref
		coherence = np.where(a != 0, np.abs(cs_avg)**2 / a, 0)
		phase_err = np.sqrt(np.where(coherence != 0, (1 - coherence) / \
			(2 * coherence * n * M), 0))

	return phase_err
## End of function 'get_phase_err'


################################################################################
def phase_to_tlags(phase, f):
	"""
			phase_to_tlags
	
	Converts a complex phase (in radians) to a time lag (in seconds).
	
	"""
	assert np.shape(phase) == np.shape(f), "ERROR: Phase array must have same \
dimensions as frequency array."

	with np.errstate(all='ignore'):
		tlags =  np.where(f != 0, phase / (2.0 * np.pi * f), 0)
		
	return tlags
## End of function 'phase_to_tlags'


################################################################################
def plot_lag_freq(out_root, plot_ext, prefix, freq, phase, err_phase, tlag, \
	err_tlag, lo_freq, up_freq, lo_energy, up_energy):
	"""
			plot_lag_freq
	
	Plots the lag-frequency spectrum for a given energy range.
	
	"""
	
	font_prop = font_manager.FontProperties(size=16)
	
	plot_file = out_root + "_lag-freq." + plot_ext
	print "Lag-frequency spectrum: %s" % plot_file

	fig, ax = plt.subplots(1,1)
	ax.plot([freq[0], freq[-1]],[0,0], lw=1.5, ls='dashed', c='black')
# 	ax.plot([freq[0], freq[-1]],[np.pi,np.pi], lw=1.5, ls='dashed', c='black')
# 	ax.plot([freq[0], freq[-1]],[-np.pi,-np.pi], lw=1.5, ls='dashed', c='black')
# 	ax.errorbar(freq, phase, yerr=phase_err, lw=3, c='blue', \
# 		ls='steps-mid', elinewidth=2, capsize=2)
	ax.errorbar(freq, tlag, yerr=err_tlag, lw=2, c='blue', \
		ls='steps-mid', capsize=2, elinewidth=2)
	ax.set_xlabel('Frequency (Hz)', fontproperties=font_prop)
	ax.set_ylabel('Time lag (s)', fontproperties=font_prop)
# 	ax.set_ylabel('Phase lag (radians)', fontproperties=font_prop)
	ax.set_xlim(lo_freq, up_freq)
# 	ax.set_ylim(-0.3, 0.3)
# 	ax.set_ylim(-6, 6)
	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	ax.set_title("Lag-frequency spectrum, %s, channels %d - %d" % (prefix, \
		lo_energy, up_energy), fontproperties=font_prop)
	
	fig.set_tight_layout(True)
	plt.savefig(plot_file, dpi=160)
# 		plt.show()
	plt.close()
# 	subprocess.call(['open', plot_file])
		
## End of function 'plot_lag_freq'


################################################################################
def plot_lag_energy(out_root, plot_ext, prefix, phase, err_phase, tlag, \
	err_tlag, lo_freq, up_freq, detchans):
	"""
			plot_lag_energy

	Plots the lag-energy spectrum for a given frequency range.
	
	"""
	font_prop = font_manager.FontProperties(size=16)

	
	plot_file = out_root + "_lag-energy." + plot_ext
	print "Lag-energy spectrum: %s" % plot_file
	
	e_chans = np.arange(0, detchans)
	
	## Deleting the values at energy channel 10	
	phase = np.delete(phase, 10)
	err_phase = np.delete(err_phase, 10)
	tlag = np.delete(tlag, 10)
	err_tlag = np.delete(err_tlag, 10)
	e_chans = np.delete(e_chans, 10)
	
	fig, ax = plt.subplots(1,1)
	ax.plot([0,detchans],[0,0], lw=1.5, ls='dashed', c='black')
# 	ax.plot([0,detchans],[np.pi,np.pi], lw=1.5, ls='dashed', c='black')
# 	ax.plot([0,detchans],[-np.pi,-np.pi], lw=1.5, ls='dashed', c='black')
# 	ax.errorbar(e_chans, phase, yerr=err_phase, lw=3, c='red', \
# 		ls="steps-mid", elinewidth=2, capsize=2)
	ax.errorbar(e_chans, tlag, yerr=err_tlag, ls='none', marker='x', \
		ms=10, mew=2, mec='red', ecolor='red', \
		elinewidth=2, capsize=2)
	ax.set_xlabel('Energy channel (0-%d)' % (detchans-1), \
		fontproperties=font_prop)
	ax.set_ylabel('Time lag (s)', fontproperties=font_prop)
# 	ax.set_ylabel('Phase lag (radians)', fontproperties=font_prop)
	ax.set_xlim(1.5,25.5)
	ax.set_ylim(1.30*np.min(tlag[2:25]), 1.30*np.max(tlag[2:25]))
# 	ax.set_ylim(-0.03, 0.03)
# 	ax.set_ylim(-np.pi, np.pi)
# 	ax.set_ylim(-6, 6)
	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	ax.set_title("Lag-energy spectrum, %s, %.2f - %.2f Hz" % (prefix, lo_freq,\
		up_freq), fontproperties=font_prop)
	
	fig.set_tight_layout(True)
	plt.savefig(plot_file, dpi=160)
# 	plt.show()
	plt.close()
	subprocess.call(['open', plot_file])
	
## End of function 'plot_lag_energy'


################################################################################
def main(in_file, out_file, plot_root, plot_ext, prefix, lo_freq, up_freq, \
	lo_energy, up_energy):
	"""
			main
			
	Computes the phase lag and time lag from the average cross spectrum. Note
	that power_ci, power_ref, and cs_avg should be unnormalized and without
	noise subtracted.
	
	"""
	
	## Get necessary information and data from the input file
	freq, cs_avg, power_ci, power_ref, dt, n_bins, detchans, num_seconds, \
		num_seg, mean_rate_ci, mean_rate_ref, evt_list = get_inputs(in_file)
	
	## If cross spectrum contains positive and negative frequencies, only keep
	## the positive ones
	if np.shape(cs_avg) == (n_bins, detchans):
		nyq_ind = np.argmax(freq)+1  ## because in python, the scipy fft makes
			## the nyquist frequency negative, and we want it to be positive! 
			## (it is actually both pos and neg)
		freq = np.abs(freq[0:nyq_ind + 1])  ## because it slices at end-1, and 
			## we want to include 'nyq_ind'; abs is because the nyquist freq is 
			## both pos and neg, and we want it pos here.
		cs_avg = cs_avg[0:nyq_ind + 1, ]
		power_ci = power_ci[0:nyq_ind + 1, ]
		power_ref = power_ref[0:nyq_ind + 1]
	
	assert np.shape(power_ci) == (n_bins/2+1, detchans)
	assert np.shape(power_ref) == (n_bins/2+1, )
	assert np.shape(cs_avg) == (n_bins/2+1, detchans)
	assert np.shape(freq) == (n_bins/2+1, )
	
	###########################
	## Averaging over energies
	###########################
	
	e_span = up_energy - lo_energy + 1  ## including both ends
	erange_cs = np.mean(cs_avg[:,lo_energy:up_energy+1], axis=1)
	erange_pow_ci = np.mean(power_ci[:,lo_energy:up_energy+1], axis=1)
	erange_pow_ref = power_ref
	
	################################################
	## Getting lag and error for lag-frequency plot
	################################################
	
	f_phase = -np.arctan2(erange_cs.imag, erange_cs.real) ## Negative sign is 
		## so that a positive lag is a hard energy lag
	f_err_phase = get_phase_err(erange_cs, erange_pow_ci, erange_pow_ref, \
		e_span, num_seg)
	f_tlag = phase_to_tlags(f_phase, freq)
	f_err_tlag = phase_to_tlags(f_err_phase, freq)
	
	##############################
	## Averaging over frequencies
	##############################
	
	f_span_low = np.argmax(freq == lo_freq)
	f_span_hi = np.argmax(freq == up_freq)
	f_span = f_span_hi - f_span_low + 1  ## including both ends
	frange_freq = freq[f_span_low:f_span_hi+1]
	frange_cs = np.mean(cs_avg[f_span_low:f_span_hi+1, ], axis=0)
	frange_pow_ci = np.mean(power_ci[f_span_low:f_span_hi+1, ], axis=0)
	frange_pow_ref = np.repeat(np.mean(power_ref[f_span_low:f_span_hi+1]), \
		detchans)
	
	#############################################
	## Getting lag and error for lag-energy plot 
	#############################################
	
	e_phase = -np.arctan2(frange_cs.imag, frange_cs.real) ## Negative sign is 
		## so that a positive lag is a hard energy lag
	e_err_phase = get_phase_err(frange_cs, frange_pow_ci, frange_pow_ref, \
		f_span, num_seg)
	f = np.repeat(np.mean(frange_freq), detchans)
	e_tlag = phase_to_tlags(e_phase, f)
	e_err_tlag = phase_to_tlags(e_err_phase, f)
	
	##########
	## Output
	##########
	
	fits_out(out_file, in_file, evt_list, dt, n_bins, num_seg, detchans, \
		lo_freq, up_freq, lo_energy, up_energy, mean_rate_ci, mean_rate_ref, \
		freq, f_phase, f_err_phase, f_tlag, f_err_tlag, e_phase, e_err_phase, \
		e_tlag, e_err_tlag)
	
	############
	## Plotting
	############
	
	plot_lag_freq(plot_root, plot_ext, prefix, freq, f_phase, f_err_phase, \
		f_tlag, f_err_tlag, lo_freq, up_freq, lo_energy, up_energy)
		
	plot_lag_energy(plot_root, plot_ext, prefix, e_phase, e_err_phase, e_tlag, \
		e_err_tlag, lo_freq, up_freq, detchans)
		
## End of function 'main'


################################################################################
if __name__ == "__main__":

	##############################################
	## Parsing input arguments and calling 'main'
	##############################################
	
	parser = argparse.ArgumentParser(usage="python get_lags.py infile outfile \
[-o plot_root] [-p prefix] [-e plot_ext] [--lf lo_freq] [--uf up_freq] \
[--le lo_energy] [--ue up_energy]", description="Computes the phase lag and \
time lag of channels of interest with a reference band from the average cross \
spectrum.", epilog="For optional arguments, default values are given in \
brackets at end of description.")

	parser.add_argument('infile', help='Name of the FITS file containing the \
cross spectrum, power spectrum of the channels of interest, and power spectrum \
of the reference band.')

	parser.add_argument('outfile', help='Name of the FITS file to write the lag\
 spectra to.')
 
 	parser.add_argument('-o', dest='plot_root', default="./plot", help="Root \
name for plots generated, to be appended with '_lag-freq.(extension)' and '_lag\
-energy.(extension)'. [./plot]")
 	
 	parser.add_argument('-p', '--prefix', dest="prefix", default="--", \
help="The identifying prefix of the data (object nickname or proposal ID). \
[--]")

	parser.add_argument('-e', '--ext', dest='plot_ext', default='png', \
help="File extension for the plots. Do not include the '.' [png]")
 	
 	parser.add_argument('--lf', dest='lo_freq', default=1.0, \
type=tools.type_positive_float,  help="The lower limit of the frequency range \
for the lag-energy spectrum to be computed for, in Hz. [1.0]")
 	
 	parser.add_argument('--uf', dest='up_freq', default=10.0, \
type=tools.type_positive_float,  help="The upper limit of the frequency range \
for the lag-energy spectrum to be computed for, in Hz. [10.0]")
 	
 	parser.add_argument('--le', dest='lo_energy', default=3, \
type=tools.type_positive_int, help="The lower limit of the energy range for the\
 lag-frequency spectrum to be computed for, in detector channels. [3]")
 	
 	parser.add_argument('--ue', dest='up_energy', type=tools.type_positive_int,\
default=20, help="The upper limit of the energy range for the lag-frequency \
spectrum to be computed for, in detector channels. [20]")
 	
	args = parser.parse_args()
	
	main(args.infile, args.outfile, args.plot_root, args.prefix, args.plot_ext,\
		args.lo_freq, args.up_freq, args.lo_energy, args.up_energy)
	
	
################################################################################

	