# retrospectr

[![Coverage badge](https://github.com/alecksphillips/retrospectr/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/alecksphillips/retrospectr/tree/python-coverage-comment-action-data)


Importance weight calculation for Stan model results with new data.


Based on an R package which can be found [here](https://github.com/codatmo/stanIncrementalImportanceSampling)

## Requirements
```
bridgestan
numpy
```


## Installation
retrospectr can be installed with either:
```bash
git clone https://github.com/alecksphillips/retrospectr.git
pip install ./retrospectr
```
or
```bash
pip install -U git+https://github.com/alecksphillips/retrospectr.git
```

## Example
```python
import cmdstanpy
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from retrospectr.importance_weights import calculate_log_weights, extract_samples
from retrospectr.resampling import resample


model_file = "test/test_models/bernoulli/bernoulli.stan"
stan_model = cmdstanpy.CmdStanModel(stan_file=model_file)

original_data = {
    "N": 10,
    "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
}
original_fit = stan_model.sample(data=original_data, chains=1)

new_data = {
    "N": 20,
    "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
}

new_fit = stan_model.sample(data=new_data, chains=1)

original_samples = extract_samples(original_fit)

new_samples = extract_samples(new_fit)

log_weights = calculate_log_weights(model_file, original_samples, original_data, new_data)

resampled_original_samples = resample(original_samples, log_weights)

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

```

![plot of example with resampled iterations](https://github.com/alecksphillips/retrospectr/blob/main/example.png?raw=true)
