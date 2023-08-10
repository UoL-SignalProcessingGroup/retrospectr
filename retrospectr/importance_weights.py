import bridgestan as bs
import numpy as np
import cmdstanpy


# def calculate_log_weights(model, old_samples, old_data, new_data):
"""

"""
#     #check old samples match the model+old_data

#     #check model+new_data matches the model+old_data

#     #calculate logProb

#     #calculate log weights

#     return log_weights


def evaluate_logProb(model, samples):
    """
    Evaluate model at each sample point in samples,
    returning logprob values
    """
    if not isinstance(model, bs.model.StanModel):
        raise TypeError("'model' is not of type ", str(bs.model.StanModel))

    unc_samples = model.param_unconstrain(samples)
    logProb = model.log_density(unc_samples, jacobian=T)
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
    return d == sample_d


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
    return old_names == new_names
