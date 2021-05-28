from UQpy.SampleMethods import LHS
from UQpy.Distributions import *
import numpy as np
import pytest

distribution = Uniform(1, 1)
distribution1 = Uniform(1, 2)
distribution2 = Uniform(3, 4)


def test_lhs_single_distribution():
    latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1, verbose=True)
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.208511], [1.86016225]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_lhs_single_distribution_list():
    latin_hypercube_sampling = LHS(dist_object=[distribution], nsamples=2, random_state=1, verbose=True)
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.208511], [1.86016225]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_lhs_joint_distribution():
    joint_distribution = JointInd(marginals=[distribution1, distribution2])
    latin_hypercube_sampling = LHS(dist_object=joint_distribution, nsamples=2, random_state=1, verbose=True)
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.417022], [5.60466515], [2.72032449], [3.00022875]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_lhs_random_criterion():
    latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1,
                                   verbose=True, criterion='random')
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.208511], [1.86016225]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_lhs_centered_criterion():
    latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1,
                                   verbose=True, criterion='centered')
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.25], [1.75]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_lhs_maxmin_criterion():
    latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1,
                                   verbose=True, criterion='maximin')
    actual_samples = latin_hypercube_sampling.samples.flatten()
    expected_samples = np.array([[1.208511], [1.86016225]]).flatten()
    np.testing.assert_allclose(expected_samples, actual_samples, rtol=1e-6)


def test_lhs_wrong_criterion():
    with pytest.raises(NotImplementedError):
        latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1,
                                       verbose=True, criterion='wrong')


def test_lhs_wrong_samples_number():
    with pytest.raises(ValueError):
        latin_hypercube_sampling = LHS(dist_object=distribution, random_state=1, nsamples=1.1, verbose=True)


def test_lhs_wrong_distribution_type_list():
    with pytest.raises(TypeError):
        latin_hypercube_sampling = LHS(dist_object=[np.array([0])], nsamples=2, random_state=1, verbose=True)


def test_lhs_wrong_distribution_type():
    with pytest.raises(TypeError):
        latin_hypercube_sampling = LHS(dist_object=np.array([0]), nsamples=2, random_state=1, verbose=True)


def test_lhs_wrong_random_state():
    with pytest.raises(TypeError):
        latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1.1, verbose=True)


def test_lhs_max_min_metric_error_1():
    latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1, verbose=True)
    samples = np.array([[0.208511], [0.86016225]])
    with pytest.raises(NotImplementedError):
        latin_hypercube_sampling.max_min(samples=samples, metric='wrong')


def test_lhs_max_min_metric_error_2():
    latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1, verbose=True)
    samples = np.array([[0.208511], [0.86016225]])
    with pytest.raises(ValueError):
        latin_hypercube_sampling.max_min(samples=samples, metric=0)


def test_lhs_max_min_iterations_error():
    latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1, verbose=True)
    samples = np.array([[0.208511], [0.86016225]])
    with pytest.raises(ValueError):
        latin_hypercube_sampling.max_min(samples=samples, iterations=1.1)


def test_lhs_max_min_iterations_error():
    latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1, verbose=True)
    samples = np.array([[0.208511], [0.86016225]])
    with pytest.raises(ValueError):
        latin_hypercube_sampling.max_min(samples=samples, iterations=1.1)


def test_lhs_correlate_iterations_error():
    latin_hypercube_sampling = LHS(dist_object=distribution, nsamples=2, random_state=1, verbose=True)
    samples = np.array([[2.08511002e-01, 5.71874087e-05], [8.60162247e-01, 6.51166286e-01]])
    actual_samples = latin_hypercube_sampling\
        .correlate(samples=samples, iterations=1, random_state=np.random.RandomState(1))
    expected_samples = np.array([[2.08511002e-01, 5.71874087e-05], [8.60162247e-01, 6.51166286e-01]])
    np.testing.assert_allclose(expected_samples, actual_samples, atol=1e-6)