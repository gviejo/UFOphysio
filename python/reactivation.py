import numpy as np


def compute_reactivation(grp_spikes, new_wake_ep, bin_size):
    """Eigen-decompose the z-scored wake population correlation matrix and
    return the significant (Marchenko-Pastur thresholded) eigenvectors/eigenvalues."""
    wake_count = grp_spikes.count(bin_size, new_wake_ep)
    wake_count = wake_count.smooth(3 * bin_size)
    wake_count = wake_count - wake_count.mean(0)
    wake_count = wake_count / wake_count.std(0)

    Corr = np.corrcoef(wake_count.d.T)
    evals, evecs = np.linalg.eigh(Corr)
    q = wake_count.shape[1] / wake_count.shape[0]
    lambda_plus = (1 + np.sqrt(q)) ** 2
    idx = np.where(evals > lambda_plus)[0]
    significant_evecs = evecs[:, idx]
    significant_evals = evals[idx]
    return significant_evecs, significant_evals