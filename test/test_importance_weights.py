from pathlib import Path
import os
import pytest
import bridgestan as bs
import cmdstanpy
import numpy as np
import json

from retrospectr.importance_weights import (
    calculate_log_weights,
    evaluate_logProb,
    extract_samples,
    check_sample_dim,
    check_models
)

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
def eight_schools_logProbs():
    return np.load(os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools_logProbs.npy'
    ))


@pytest.fixture
def eight_schools_log_weights():
    return np.load(os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools_log_weights.npy'
    ))


@pytest.fixture
def eight_schools_new_data_file():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools.new_data.json'
    )


@pytest.fixture
def eight_schools_new_data_json(eight_schools_new_data_file):
    with open(eight_schools_new_data_file) as f:
        json_data = f.read()
    return json_data


@pytest.fixture
def eight_schools_new_data_dict(eight_schools_new_data_file):
    with open(eight_schools_new_data_file) as f:
        json_dict = json.load(f)
    return json_dict


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


@pytest.fixture
def eight_schools_bs_model(eight_schools_model_file, eight_schools_data_file):
    return bs.StanModel.from_stan_file(eight_schools_model_file, model_data=eight_schools_data_file)


@pytest.fixture
def eight_schools_new_bs_model(eight_schools_model_file, eight_schools_new_data_file):
    return bs.StanModel.from_stan_file(eight_schools_model_file, model_data=eight_schools_new_data_file)


@pytest.fixture
def seven_schools_bs_model(eight_schools_model_file, seven_schools_data_file):
    return bs.StanModel.from_stan_file(eight_schools_model_file, model_data=seven_schools_data_file)


@pytest.fixture
def othermodel_data():
    return os.path.join(
        TEST_MODELS_PATH, 'bernoulli', 'bernoulli.data.json'
    )


@pytest.fixture
def othermodel_samples():
    return np.load(
        TEST_MODELS_PATH, 'bernoulli', 'bernoulli_samples.npy'
    )


@pytest.fixture
def invalid_model():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools.data.json'
    )


class TestCalculateLogWeights:
    def test_good(self, eight_schools_model_file, eight_schools_samples, eight_schools_data_file, eight_schools_new_data_file,
                  eight_schools_log_weights):
        log_weights = calculate_log_weights(
            eight_schools_model_file, eight_schools_samples,
            eight_schools_data_file, eight_schools_new_data_file)
        np.testing.assert_almost_equal(log_weights, eight_schools_log_weights)

    def test_good_json_string_data(self, eight_schools_model_file, eight_schools_samples, eight_schools_data_json,
                                   eight_schools_new_data_json, eight_schools_log_weights):
        log_weights = calculate_log_weights(
            eight_schools_model_file, eight_schools_samples,
            eight_schools_data_json, eight_schools_new_data_json)
        np.testing.assert_almost_equal(log_weights, eight_schools_log_weights)

    def test_good_python_dict_data(self, eight_schools_model_file, eight_schools_samples, eight_schools_data_dict,
                                   eight_schools_new_data_dict, eight_schools_log_weights):
        log_weights = calculate_log_weights(
            eight_schools_model_file, eight_schools_samples,
            eight_schools_data_dict, eight_schools_new_data_dict)
        np.testing.assert_almost_equal(log_weights, eight_schools_log_weights)

    def test_invalid_old_data(self, eight_schools_model_file, eight_schools_samples, eight_schools_bad_data_file,
                              eight_schools_new_data_file):
        # Should get RuntimeError from bridgestan
        with np.testing.assert_raises(RuntimeError):
            calculate_log_weights(
              eight_schools_model_file, eight_schools_samples,
              eight_schools_bad_data_file, eight_schools_new_data_file)

    def test_invalid_new_data(self, eight_schools_model_file, eight_schools_samples, eight_schools_data_file,
                              eight_schools_bad_data_file):
        # Should get RuntimeError from bridgestan
        with np.testing.assert_raises(RuntimeError):
            calculate_log_weights(
              eight_schools_model_file, eight_schools_samples,
              eight_schools_data_file, eight_schools_bad_data_file)

    def test_invalid_stan_model(self, invalid_model, eight_schools_samples, eight_schools_data_file,
                                eight_schools_new_data_file):
        with np.testing.assert_raises(ValueError):
            calculate_log_weights(
              invalid_model, eight_schools_samples,
              eight_schools_data_file, eight_schools_data_file)

    def test_invalid_samples(self, invalid_model, seven_schools_samples, eight_schools_data_file, eight_schools_new_data_file):
        with np.testing.assert_raises(ValueError):
            calculate_log_weights(
              invalid_model, seven_schools_samples,
              eight_schools_data_file, eight_schools_new_data_file)


class TestEvaluateLogProb():
    def test_good_single(self, eight_schools_bs_model, eight_schools_samples, eight_schools_logProbs):
        samples = eight_schools_samples[0, 0, :]
        log_prob = evaluate_logProb(eight_schools_bs_model, samples)
        np.testing.assert_almost_equal(log_prob, eight_schools_logProbs[0, 0])

    def test_good_array(self, eight_schools_bs_model, eight_schools_samples, eight_schools_logProbs):
        samples = np.array(eight_schools_samples)
        log_prob = evaluate_logProb(eight_schools_bs_model, samples)
        np.testing.assert_almost_equal(log_prob, eight_schools_logProbs)

    def test_invalid_model(self, invalid_model, eight_schools_samples):
        with np.testing.assert_raises(TypeError):
            evaluate_logProb(invalid_model, eight_schools_samples)

    def test_invalid_samples(self, eight_schools_bs_model, invalid_model):
        with np.testing.assert_raises(TypeError):
            evaluate_logProb(eight_schools_bs_model, invalid_model)


class TestExtractSamples:
    def test_match(self, eight_schools_cmdstanpy_fit, eight_schools_samples):
        extracted_samples = extract_samples(eight_schools_cmdstanpy_fit)
        np.testing.assert_array_equal(extracted_samples, eight_schools_samples)

    def test_invalid_input(self, invalid_model):
        with np.testing.assert_raises(TypeError):
            extract_samples(invalid_model)


class TestCheckSampleDim:
    def test_good(self, eight_schools_bs_model, eight_schools_samples):
        sample_check = check_sample_dim(eight_schools_bs_model, eight_schools_samples)
        assert sample_check

    def test_bad(self, eight_schools_bs_model, seven_schools_samples):
        with np.testing.assert_raises(ValueError):
            check_sample_dim(eight_schools_bs_model, seven_schools_samples)

    def test_invalid_input_a(self, invalid_model, eight_schools_samples):
        with np.testing.assert_raises(TypeError):
            check_sample_dim(invalid_model, eight_schools_samples)

    def test_invalid_input_b(self, eight_schools_bs_model, invalid_model):
        with np.testing.assert_raises(TypeError):
            check_sample_dim(eight_schools_bs_model, invalid_model)


class TestCheckModels:
    def test_good(self, eight_schools_bs_model, eight_schools_new_bs_model):
        model_check = check_models(eight_schools_bs_model, eight_schools_new_bs_model)
        assert model_check

    def test_bad(self, eight_schools_bs_model, seven_schools_bs_model):
        with np.testing.assert_raises(ValueError):
            check_models(eight_schools_bs_model, seven_schools_bs_model)

    def test_invalid_input_a(self, invalid_model, eight_schools_bs_model):
        with np.testing.assert_raises(TypeError):
            check_models(invalid_model, eight_schools_bs_model)

    def test_invalid_input_b(self, eight_schools_bs_model, invalid_model):
        with np.testing.assert_raises(TypeError):
            check_models(eight_schools_bs_model, invalid_model)
