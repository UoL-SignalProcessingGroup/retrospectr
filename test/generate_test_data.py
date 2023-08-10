from retrospectr.importance_weights import evaluate_logProb
from pathlib import Path
import os
from cmdstanpy import CmdStanModel
import numpy as np


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
