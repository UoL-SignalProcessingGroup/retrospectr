import bridgestan as bs
import numpy as np
import cmdstanpy
from scipy.special import logsumexp


def calculate_log_weights(model, old_samples, old_data, new_data):
    """
    Calculate importance weights of old_samples generated by model with old_data
    reweighted with new_data.

    param: model: Path to .stan file containing the model
    param: old_samples: numpy array of samples from model with original/old data
    param: old_data: Path to json file containing old data
    param: new_data: Path to json file containing new data

    returns: log importance weights of old_samples conditioned on new data
    """
    # check old samples match the model+old_data
    old_model = bs.StanModel.from_stan_file(model, model_data=old_data)
    check_sample_dim(old_model, old_samples)

    # check model+new_data matches the model+old_data
    new_model = bs.StanModel.from_stan_file(model, model_data=new_data)
    check_models(old_model, new_model)

    # calculate logProbs
    logProb_old = evaluate_logProb(old_model, old_samples)
    logProb_new = evaluate_logProb(new_model, old_samples)

    log_weights = logProb_new - logProb_old
    log_weights = log_weights - logsumexp(log_weights)

    return log_weights


def evaluate_logProb(model, samples):
    """
    Evaluate model at each sample point in samples,
    returning logprob values
    """
    if not isinstance(model, bs.model.StanModel):
        raise TypeError("'model' is not of type ", str(bs.model.StanModel))

    if not isinstance(samples, np.ndarray):
        raise TypeError("'samples' is not of type ", str(np.ndarray))

    if samples.ndim == 1:
        # Single iteration
        samples = samples.reshape((1, 1, samples.shape[0]))

    unc_samples = np.array([[
        model.param_unconstrain(np.array(samples[iter, chain, :])) for chain in range(samples.shape[1])]
        for iter in range(samples.shape[0])])

    logProb = np.array([[
        model.log_density(np.array(unc_samples[iter, chain, :])) for chain in range(unc_samples.shape[1])]
        for iter in range(unc_samples.shape[0])])
    return logProb


def extract_samples(fit):
    """
    Extract parameter samples from CmdstanMCMC fit.
    Excludes Stan diagnostic parameters.
    """
    if not isinstance(fit, cmdstanpy.CmdStanMCMC):
        raise TypeError("'fit' is not of type ", str(cmdstanpy.CmdStanMCMC))

    is_param = [not (x.endswith('__')) for x in fit.column_names]
    return fit.draws()[:, :, is_param]


def check_sample_dim(model, samples) -> bool:
    """
    Check that samples (e.g. from a Stan model) are of the same dimension as a
    bridgestan model. Assumes samples contains constrained parameters.
    """
    if not isinstance(model, bs.model.StanModel):
        raise TypeError("'model' is not of type ", str(bs.model.StanModel))
    if not isinstance(samples, np.ndarray):
        raise TypeError("'samples' is not of type ", str(np.ndarray))

    d = model.param_num()
    sample_d = samples.shape[2]
    check = d == sample_d
    if check is False:
        raise ValueError("Dimension of model with old_data does not match dimension of samples (", d, " != ", sample_d)
    return check


def check_models(model, new_model) -> bool:
    """
    Check that two  bridgestan models are of the same dimension and relate to
    the same parameters. These models need not have equal data.
    """
    if not isinstance(model, bs.model.StanModel):
        raise TypeError("'model' is not of type ", str(bs.model.StanModel))
    if not isinstance(new_model, bs.model.StanModel):
        raise TypeError("'new_model' is not of type ", str(bs.model.StanModel))
    print("type:", str(type(model)))
    # print("type:", str(type(new_model)))

    old_names = model.param_names()
    new_names = new_model.param_names()
    check = old_names == new_names
    if check is False:
        raise ValueError("Models + dimensions do not match")
    return check
