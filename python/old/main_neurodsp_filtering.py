import numpy as np
from pylab import *

from neurodsp.filt import filter_signal

from neurodsp.sim import sim_combined
from neurodsp.utils import create_times

from neurodsp.plts.time_series import plot_time_series

# Set the random seed, for consistency simulating data
np.random.seed(0)

fs = 100
n_seconds = 3
times = create_times(n_seconds, fs)
components = {'sim_powerlaw' : {'exponent' : 0},
              'sim_oscillation' : [{'freq' : 6}, {'freq' : 1}]}

variances = [0.1, 1, 1]

sig = sim_combined(n_seconds, fs, components, variances)

sig[:fs] = 0

f_range = (4, 8)

sig_filt_short = filter_signal(sig, fs, 'bandpass', f_range, n_seconds=.1)
sig_filt_long = filter_signal(sig, fs, 'bandpass', f_range, n_seconds=1)

# Plot filtered signal
plot_time_series(times, [sig, sig_filt_short, sig_filt_long],
                 ['Raw', 'Short Filter', 'Long Filter'])


# Visualize the kernels and frequency responses
print('Short filter')
sig_filt_short = filter_signal(sig, fs, 'bandpass', f_range, n_seconds=.1,
                               plot_properties=True)
print('\n\nLong filter')
sig_filt_long = filter_signal(sig, fs, 'bandpass', f_range, n_seconds=1,
                              plot_properties=True)

show()