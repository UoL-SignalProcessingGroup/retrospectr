import numpy as np
import cmdstanpy
from retrospectr.importance_weights import extract_samples


def resample(samples, log_weights, seed=0):

    if isinstance(samples, cmdstanpy.CmdStanMCMC):
        samples = extract_samples(samples)

    rng = np.random.default_rng(seed=seed)
    niters = log_weights.shape[0]
    nchains = log_weights.shape[1]
    nparams = samples.shape[2]

    nsamples = niters*nchains
    flat_log_weights = log_weights.reshape((nsamples))

    resampled_iterations = rng.choice(
        nsamples,
        size=nsamples,
        p=np.exp(flat_log_weights))

    flat_samples = samples.reshape(nsamples, 1, nparams)
    resampled_samples = flat_samples[resampled_iterations, :]
    return resampled_samples
