from pathlib import Path
import os
import pytest
import bridgestan as bs
import cmdstanpy
import numpy as np

from retrospectr.importance_weights import (
    # calculate_log_weights,
    evaluate_logProb,
    extract_samples,
    check_sample_dim,
    check_models
)

# TEST_MODELS_PATH = os.path.join(Path(__file__).parent, 'test_models')
TEST_MODELS_PATH = os.path.join('test', 'test_models')


@pytest.fixture
def eight_schools_model_file():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools.stan'
    )


@pytest.fixture
def eight_schools_data():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools.data.json'
    )


@pytest.fixture
def eight_schools_samples():
    return np.load(os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools_samples.npy'
    ))


@pytest.fixture
def eight_schools_new_data():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools.new_data.json'
    )


@pytest.fixture
def seven_schools_data():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'seven_schools.data.json'
    )


@pytest.fixture
def seven_schools_samples():
    return np.load(os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'seven_schools_samples.npy'
    ))


@pytest.fixture
def eight_schools_bad_data():
    return os.path.join(
        TEST_MODELS_PATH, 'eight_schools', 'eight_schools.bad_data.json'
    )


@pytest.fixture
def eight_schools_cmdstanpy_fit(eight_schools_model_file, eight_schools_data):
    model = cmdstanpy.CmdStanModel(stan_file=eight_schools_model_file)
    fit = model.sample(data=eight_schools_data, chains=1, iter_sampling=200, iter_warmup=200, seed=0)
    return fit


@pytest.fixture
def eight_schools_bs_model(eight_schools_model_file, eight_schools_data):
    return bs.StanModel.from_stan_file(eight_schools_model_file, model_data=eight_schools_data)


@pytest.fixture
def eight_schools_new_bs_model(eight_schools_model_file, eight_schools_new_data):
    return bs.StanModel.from_stan_file(eight_schools_model_file, model_data=eight_schools_new_data)


@pytest.fixture
def seven_schools_bs_model(eight_schools_model_file, seven_schools_data):
    return bs.StanModel.from_stan_file(eight_schools_model_file, model_data=seven_schools_data)


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


# class TestCalculateLogWeights:
#     def test_good(eight_schools, eight_schools_samples, eight_schools_data, eight_schools_new_data):
#       np.testing.assert_no_warnings(
#           log_weights = calculate_log_weights(
#               eight_schools, eight_schools_samples,
#               eight_schools_data, eight_schools_new_data)
#       )

#     def test_invalid_data():


#     def test_invalid_stan_model():


#     def test_invalid_samples():


class TestEvaluateLogProb():
    def test_good(self, eight_schools_bs_model, eight_schools_samples):
        with np.testing.assert_no_warnings():
            evaluate_logProb(eight_schools_bs_model, eight_schools_samples)

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
        assert sample_check == True

    def test_bad(self, eight_schools_bs_model, seven_schools_samples):
        sample_check = check_sample_dim(eight_schools_bs_model, seven_schools_samples)
        assert sample_check == False

    def test_invalid_input_a(self, invalid_model, eight_schools_samples):
        with np.testing.assert_raises(TypeError):
            check_sample_dim(invalid_model, eight_schools_samples)

    def test_invalid_input_b(self, eight_schools_bs_model, invalid_model):
        with np.testing.assert_raises(TypeError):
            check_sample_dim(eight_schools_bs_model, invalid_model)


class TestCheckModels:
    def test_good(self, eight_schools_bs_model, eight_schools_new_bs_model):
        model_check = check_models(eight_schools_bs_model, eight_schools_new_bs_model)
        assert model_check == True

    def test_bad(self, eight_schools_bs_model, seven_schools_bs_model):
        model_check = check_models(eight_schools_bs_model, seven_schools_bs_model)
        assert model_check == False

    def test_invalid_input_a(self, invalid_model, eight_schools_bs_model):
        with np.testing.assert_raises(TypeError):
            check_models(invalid_model, eight_schools_bs_model)

    def test_invalid_input_b(self, eight_schools_bs_model, invalid_model):
        with np.testing.assert_raises(TypeError):
            check_models(eight_schools_bs_model, invalid_model)
