from pathlib import Path
import os
import pytest
import cmdstanpy
import numpy as np
import json

from retrospectr.resampling import resample

TEST_MODELS_PATH = os.path.join(Path(__file__).parent, 'test_models')


@pytest.fixture
def eight_schools_model_file():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools.stan'
    )


@pytest.fixture
def eight_schools_data_file():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools.data.json'
    )


@pytest.fixture
def eight_schools_data_json(eight_schools_data_file):
    with open(eight_schools_data_file) as f:
        json_data = f.read()
    return json_data


@pytest.fixture
def eight_schools_data_dict(eight_schools_data_file):
    with open(eight_schools_data_file) as f:
        json_dict = json.load(f)
    return json_dict


@pytest.fixture
def eight_schools_samples():
    return np.load(os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools_samples.npy'
    ))


@pytest.fixture
def eight_schools_log_weights():
    return np.load(os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools_log_weights.npy'
    ))


@pytest.fixture
def eight_schools_resampled_samples():
    return np.load(os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools_resampled_samples.npy'
    ))


@pytest.fixture
def seven_schools_data_file():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'seven_schools.data.json'
    )


@pytest.fixture
def seven_schools_samples():
    return np.load(os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'seven_schools_samples.npy'
    ))


@pytest.fixture
def eight_schools_bad_data_file():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools.bad_data.json'
    )


@pytest.fixture
def eight_schools_cmdstanpy_fit(eight_schools_model_file, eight_schools_data_file):
    model = cmdstanpy.CmdStanModel(stan_file=eight_schools_model_file)
    fit = model.sample(data=eight_schools_data_file, chains=2, iter_sampling=200, iter_warmup=200, seed=0)
    return fit


class TestResampled:
    def test_good_sample_array(self, eight_schools_samples, eight_schools_log_weights, eight_schools_resampled_samples):
        resampled_samples = resample(eight_schools_samples, eight_schools_log_weights, seed=0)
        np.testing.assert_equal(resampled_samples, eight_schools_resampled_samples)

    def test_good_cmdstanpy_fit(self, eight_schools_cmdstanpy_fit, eight_schools_log_weights, eight_schools_resampled_samples):
        resampled_samples = resample(eight_schools_cmdstanpy_fit, eight_schools_log_weights, seed=0)
        np.testing.assert_equal(resampled_samples, eight_schools_resampled_samples)

    def test_bad_weights_wrong_len(self, eight_schools_samples, eight_schools_log_weights):
        tmp_log_weights = eight_schools_log_weights[0:(len(eight_schools_log_weights)-1)]
        with np.testing.assert_raises(ValueError):
            resample(eight_schools_samples, tmp_log_weights, seed=0)

    def test_bad_weights_not_sum_to_one(self, eight_schools_samples, eight_schools_log_weights):
        tmp_log_weights = eight_schools_log_weights
        tmp_log_weights[0] = 1.0
        with np.testing.assert_raises(ValueError):
            resample(eight_schools_samples, tmp_log_weights, seed=0)
