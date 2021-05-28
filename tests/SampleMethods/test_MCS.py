from UQpy.SampleMethods import MCS
from UQpy.Distributions import *
import numpy as np
import pytest

distribution = Uniform(1, 1)
distribution1 = Uniform(1, 2)
distribution2 = Uniform(3, 4)


def test_mcs_single_distribution():
    monte_carlo_sampling = MCS(dist_object=distribution, nsamples=2, random_state=1, verbose=True)
    actual_samples = monte_carlo_sampling.samples
    expected_samples = np.array([[1.417022], [1.72032449]])
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_mcs_distribution_continuous_list():
    monte_carlo_sampling = MCS(dist_object=[distribution1, distribution2], nsamples=2, random_state=1, verbose=True)
    actual_samples = monte_carlo_sampling.samples
    expected_samples = np.array([[1.83404401, 3.0004575], [2.44064899, 4.20933029]])
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_mcs_distribution_joint():
    joint_distribution = JointInd(marginals=[distribution1, distribution2])
    monte_carlo_sampling = MCS(dist_object=joint_distribution, nsamples=2, random_state=1, verbose=True)
    actual_samples = monte_carlo_sampling.samples
    expected_samples = np.array([[1.83404401, 3.0004575], [2.44064899, 4.20933029]])
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_mcs_distribution_joint_list():
    joint_distribution = JointInd(marginals=[distribution1, distribution2])
    monte_carlo_sampling = MCS(dist_object=[joint_distribution], nsamples=2, random_state=1, verbose=True)
    actual_samples = monte_carlo_sampling.samples.flatten()
    expected_samples = np.array([[1.83404401, 3.0004575], [2.44064899, 4.20933029]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_mcs_distribution_wrong_distribution_object_list():
    with pytest.raises(TypeError):
        monte_carlo_sampling = MCS(dist_object=[np.array([0])], nsamples=2, random_state=1, verbose=True)


def test_mcs_distribution_wrong_distribution_object():
    with pytest.raises(TypeError):
        monte_carlo_sampling = MCS(dist_object=np.array([0]), nsamples=2, random_state=1, verbose=True)


def test_mcs_wrong_random_state():
    with pytest.raises(TypeError):
        monte_carlo_sampling = MCS(dist_object=distribution, nsamples=2, random_state="a", verbose=True)


def test_mcs_wrong_random_state_distribution_list():
    with pytest.raises(TypeError):
        monte_carlo_sampling = MCS(dist_object=[distribution], nsamples=2, random_state="a", verbose=True)


def test_mcs_run_samples_none():
    monte_carlo_sampling = MCS(dist_object=distribution, random_state=1, verbose=True)
    with pytest.raises(ValueError):
        monte_carlo_sampling.run(nsamples=None)


def test_mcs_run_samples_value_error():
    monte_carlo_sampling = MCS(dist_object=distribution, random_state=1, verbose=True)
    with pytest.raises(ValueError):
        monte_carlo_sampling.run(nsamples=1.1)


def test_mcs_run_samples_random_state_int():
    monte_carlo_sampling = MCS(dist_object=distribution, random_state=1, verbose=True)
    monte_carlo_sampling.run(nsamples=2, random_state=1)
    actual_samples = monte_carlo_sampling.samples
    expected_samples = np.array([[1.417022], [1.72032449]])
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_mcs_run_samples_random_state_type_error():
    monte_carlo_sampling = MCS(dist_object=distribution, random_state=1, verbose=True)
    with pytest.raises(TypeError):
        monte_carlo_sampling.run(nsamples=2, random_state=np.array([0]))


def test_mcs_run_append_samples_single_distribution():
    monte_carlo_sampling = MCS(dist_object=distribution, nsamples=1, random_state=1, verbose=True)
    monte_carlo_sampling.run(nsamples=1)
    actual_samples = monte_carlo_sampling.samples
    expected_samples = np.array([[1.417022], [1.72032449]])
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_mcs_run_append_samples_list():
    monte_carlo_sampling = MCS(dist_object=[distribution], nsamples=1, random_state=1, verbose=True)
    monte_carlo_sampling.run(nsamples=1)
    actual_samples = monte_carlo_sampling.samples
    expected_samples = np.array([[1.417022], [1.72032449]])
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_mcs_transform_u01_distribution():
    monte_carlo_sampling = MCS(dist_object=distribution, nsamples=2, random_state=1, verbose=True)
    monte_carlo_sampling.transform_u01()
    actual_samples = monte_carlo_sampling.samplesU01
    expected_samples = np.array([[0.417022], [0.72032449]])
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_mcs_transform_u01_distribution_list_1():
    monte_carlo_sampling = MCS(dist_object=[distribution], nsamples=2, random_state=1, verbose=True)
    monte_carlo_sampling.transform_u01()
    actual_samples = monte_carlo_sampling.samplesU01
    expected_samples = np.array([[0.417022], [0.72032449]])
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_mcs_transform_u01_distribution_list_2():
    monte_carlo_sampling = MCS(dist_object=[distribution, Poisson(mu=1)], nsamples=2, random_state=1, verbose=True)
    monte_carlo_sampling.transform_u01()
    actual_samples = monte_carlo_sampling.samplesU01
    expected_samples = [np.array([[0.4170222], [0.3678794]]), np.array([[0.72032449], [0.36787944]])]
    np.testing.assert_allclose(expected_samples[0], actual_samples[0], rtol=1e-6)
    np.testing.assert_allclose(expected_samples[1], actual_samples[1], rtol=1e-6)
