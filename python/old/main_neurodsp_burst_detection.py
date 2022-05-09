import numpy as np
from pylab import *
from neurodsp import sim
from neurodsp.utils import create_times
from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats

from neurodsp.plts.time_series import plot_time_series, plot_bursts

# Set the random seed, for consistency simulating data
sim.set_random_seed(0)

# Simulation settings
fs = 1000
n_seconds = 5

# Define simulation components
components = {'sim_synaptic_current' : {'n_neurons':1000, 'firing_rate':2,
                                        't_ker':1.0, 'tau_r':0.002, 'tau_d':0.02},
              'sim_bursty_oscillation' : {'freq' : 10,
                                          'prob_enter_burst' : .2, 'prob_leave_burst' : .2}}

# Simulate a signal with a bursty oscillation with an aperiodic component & a time vector
sig = sim.sim_combined(n_seconds, fs, components)
times = create_times(n_seconds, fs)

# Plot the simulated data
plot_time_series(times, sig, 'Simulated EEG')


# Settings for the dual threshold algorithm
amp_dual_thresh = (1, 2)
f_range = (8, 12)

# Detect bursts using dual threshold algorithm
bursting = detect_bursts_dual_threshold(sig, fs, amp_dual_thresh, f_range)

# Plot original signal and burst activity
plot_bursts(times, sig, bursting, labels=['Simulated EEG', 'Detected Burst'])

# Compute burst statistics
burst_stats = compute_burst_stats(bursting, fs)

# Print out burst statistic information
for key, val in burst_stats.items():
    print('{:15} \t: {}'.format(key, val))




show()