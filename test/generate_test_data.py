from pathlib import Path
import os
from cmdstanpy import CmdStanModel
import numpy as np
import bridgestan as bs
from scipy.special import logsumexp

current_path = Path(__file__).parent
model_path = os.path.join(current_path, 'test_models', 'eight_schools')
stan_file = os.path.join(model_path, 'eight_schools.stan')
data_file = os.path.join(model_path, 'eight_schools.data.json')

model = CmdStanModel(stan_file=stan_file)
fit = model.sample(data=data_file, chains=2, iter_sampling=200, iter_warmup=200, seed=0)
samples = fit.draws()
is_param = [not (x.endswith('__')) for x in fit.column_names]
samples = samples[:, :, is_param]
np.save(os.path.join(model_path, "eight_schools_samples.npy"), samples)

data_file = os.path.join(model_path, 'eight_schools.new_data.json')
model = CmdStanModel(stan_file=stan_file)
fit = model.sample(data=data_file, chains=2, iter_sampling=200, iter_warmup=200, seed=0)
samples = fit.draws()
is_param = [not (x.endswith('__')) for x in fit.column_names]
samples = samples[:, :, is_param]
np.save(os.path.join(model_path, "eight_schools_new_samples.npy"), samples)

data_file = os.path.join(model_path, 'seven_schools.data.json')
fit = model.sample(data=data_file, chains=2, iter_sampling=200, iter_warmup=200, seed=0)
samples = fit.draws()
is_param = [not (x.endswith('__')) for x in fit.column_names]
samples = samples[:, :, is_param]
np.save(os.path.join(model_path, "seven_schools_samples.npy"), samples)

eight_schools_model_file = os.path.join(model_path, "eight_schools.stan")
eight_schools_data = os.path.join(model_path, "eight_schools.data.json")

model = bs.StanModel.from_stan_file(eight_schools_model_file, model_data=eight_schools_data)

samples = np.load(os.path.join(model_path, "eight_schools_samples.npy"))
unc_samples = np.array([[model.param_unconstrain(np.array(samples[iter, chain, :])) for chain in range(samples.shape[1])]
                        for iter in range(samples.shape[0])])
logProbs = np.array([[model.log_density(unc_samples[iter, chain], jacobian=True) for chain in range(samples.shape[1])]
                     for iter in range(samples.shape[0])])
np.save(os.path.join(model_path, "eight_schools_logProbs.npy"), logProbs)

new_model = bs.StanModel.from_stan_file(
    eight_schools_model_file, model_data=os.path.join(model_path, 'eight_schools.new_data.json'))
new_logProbs = np.array([[new_model.log_density(unc_samples[iter, chain], jacobian=True) for chain in range(samples.shape[1])]
                         for iter in range(samples.shape[0])])

log_weights = new_logProbs - logProbs
log_weights = log_weights - logsumexp(log_weights)
np.save(os.path.join(model_path, "eight_schools_log_weights.npy"), log_weights)

nsamples = samples.shape[0]*samples.shape[1]
tmp_samples = samples.reshape((nsamples, 1, samples.shape[2]))
tmp_log_weights = log_weights.reshape((nsamples))

rng = np.random.default_rng(seed=0)
resampled_iterations = rng.choice(
        nsamples,
        size=nsamples,
        p=np.exp(tmp_log_weights))

resampled_samples = tmp_samples[resampled_iterations, :, :]
np.save(os.path.join(model_path, "eight_schools_resampled_samples.npy"), resampled_samples)
