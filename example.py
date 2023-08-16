import cmdstanpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import json
from retrospectr.importance_weights import calculate_log_weights, extract_samples


model_file = "test/test_models/bernoulli/bernoulli.stan"
stan_model = cmdstanpy.CmdStanModel(stan_file=model_file)

original_data = {
    "N": 10,
    "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
}

original_data_file_path = "original_data.json"
with open(original_data_file_path, "w") as f:
    json.dump(original_data, f)

original_fit = stan_model.sample(data=original_data, chains=1)

new_data = {
    "N": 20,
    "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
}

new_data_file_path = "new_data.json"
with open(new_data_file_path, "w") as f:
    json.dump(new_data, f)


new_fit = stan_model.sample(data=new_data, chains=1)

original_samples = extract_samples(original_fit)

new_samples = extract_samples(new_fit)

log_weights = calculate_log_weights(model_file, original_samples, original_data_file_path, new_data_file_path)

resampled_iterations = np.random.choice(
    len(log_weights), size=len(log_weights), p=np.exp(log_weights.reshape(len(log_weights))))

resampled_original_samples = original_samples[resampled_iterations, :]

df_original = pd.DataFrame({
    "theta": original_samples.reshape(len(original_samples)),
    "model": "Original"})

df_new = pd.DataFrame({
    "theta": new_samples.reshape(len(new_samples)),
    "model": "New"})

df_resampled = pd.DataFrame({
    "theta": resampled_original_samples.reshape(len(resampled_original_samples)),
    "model": "Resampled"})


df = pd.concat((df_original, df_new, df_resampled))

seaborn.displot(df, x="theta", hue="model", kind="kde")
plt.show()
