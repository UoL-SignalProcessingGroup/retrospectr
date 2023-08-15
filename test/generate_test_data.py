from pathlib import Path
import os
from cmdstanpy import CmdStanModel
import numpy as np
import bridgestan as bs

current_path = Path(__file__).parent
model_path = os.path.join(current_path, 'test_models', 'eight_schools')
stan_file = os.path.join(model_path, 'eight_schools.stan')
data_file = os.path.join(model_path, 'eight_schools.data.json')

model = CmdStanModel(stan_file=stan_file)
fit = model.sample(data=data_file, chains=1, iter_sampling=200, iter_warmup=200, seed=0)
samples = fit.draws()
is_param = [not (x.endswith('__')) for x in fit.column_names]
samples = samples[:, :, is_param]
np.save(os.path.join(model_path, "eight_schools_samples.npy"), samples)

data_file = os.path.join(model_path, 'eight_schools.new_data.json')
model = CmdStanModel(stan_file=stan_file)
fit = model.sample(data=data_file, chains=1, iter_sampling=200, iter_warmup=200, seed=0)
samples = fit.draws()
is_param = [not (x.endswith('__')) for x in fit.column_names]
samples = samples[:, :, is_param]
np.save(os.path.join(model_path, "eight_schools_new_samples.npy"), samples)

data_file = os.path.join(model_path, 'seven_schools.data.json')
fit = model.sample(data=data_file, chains=1, iter_sampling=200, iter_warmup=200, seed=0)
samples = fit.draws()
is_param = [not (x.endswith('__')) for x in fit.column_names]
samples = samples[:, :, is_param]
np.save(os.path.join(model_path, "seven_schools_samples.npy"), samples)

eight_schools_model_file = os.path.join(model_path, "eight_schools.stan")
eight_schools_data = os.path.join(model_path, "eight_schools.data.json")

model = bs.StanModel.from_stan_file(eight_schools_model_file, model_data=eight_schools_data)

samples =  np.load(os.path.join(model_path, "eight_schools_samples.npy"))
unc_samples = [model.param_unconstrain(np.array(samples[i])) for i in range(len(samples))]
logProbs = np.array([model.log_density(unc_samples[i], jacobian=True) for i in range(len(samples))])
np.save(os.path.join(model_path, "eight_schools_logProbs.npy"), logProbs)