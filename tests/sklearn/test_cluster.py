import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from sklearn.datasets import make_blobs
from reddwarf.sklearn.cluster import BestPolisKMeans, PolisKMeans

class TestBestPolisKMeans:
    """Test suite for BestPolisKMeans class."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple clustered data for testing."""
        X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)  # type: ignore
        return X, y

    @pytest.fixture
    def complex_data(self):
        """Generate more complex clustered data."""
        X, y = make_blobs(n_samples=200, centers=5, n_features=4, random_state=123)  # type: ignore
        return X, y

    def test_initialization_default_params(self):
        """Test that BestPolisKMeans initializes with default parameters."""
        bpk = BestPolisKMeans()

        assert bpk.k_bounds == [2, 5]
        assert bpk.init == "polis"
        assert bpk.init_centers is None
        assert bpk.random_state is None
        assert bpk.best_estimator_ is None
        assert bpk.best_k_ is None
        assert bpk.best_score_ is None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        init_centers = np.array([[0, 0], [1, 1]])
        bpk = BestPolisKMeans(
            k_bounds=[3, 7],
            init="k-means++",
            init_centers=init_centers,
            random_state=42
        )
        assert bpk.init_centers is not None

        assert bpk.k_bounds == [3, 7]
        assert bpk.init == "k-means++"
        assert_array_equal(bpk.init_centers, init_centers)
        assert bpk.random_state == 42

    def test_fit_sets_attributes(self, simple_data):
        """Test that fit() properly sets the best_* attributes."""
        X, _ = simple_data
        bpk = BestPolisKMeans(k_bounds=[2, 4], random_state=42)
        bpk.fit(X)

        assert bpk.best_estimator_ is not None
        assert bpk.best_k_ is not None
        assert bpk.best_score_ is not None
        assert isinstance(bpk.best_k_, (int, np.integer))
        assert isinstance(bpk.best_score_, (float, np.floating))

    def test_best_estimator_is_fitted(self, simple_data):
        """Test that best_estimator_ is a fitted PolisKMeans instance."""
        X, _ = simple_data
        bpk = BestPolisKMeans(k_bounds=[2, 4], random_state=42)
        bpk.fit(X)

        assert isinstance(bpk.best_estimator_, PolisKMeans)
        assert hasattr(bpk.best_estimator_, 'cluster_centers_')
        assert hasattr(bpk.best_estimator_, 'labels_')
        assert bpk.best_estimator_.n_clusters == bpk.best_k_

    def test_fit_predict_returns_labels(self, simple_data):
        """Test that fit_predict() returns cluster labels."""
        X, _ = simple_data
        bpk = BestPolisKMeans(k_bounds=[2, 4], random_state=42)
        labels = bpk.fit_predict(X)

        assert labels is not None
        assert len(labels) == len(X)
        assert labels.min() >= 0
        assert labels.max() < bpk.best_k_

    def test_fit_predict_matches_labels(self, simple_data):
        """Test that fit_predict() returns same labels as best_estimator_.labels_."""
        X, _ = simple_data
        bpk = BestPolisKMeans(k_bounds=[2, 4], random_state=42)
        labels = bpk.fit_predict(X)

        assert bpk.best_estimator_ is not None
        assert labels is not None

        assert_array_equal(labels, bpk.best_estimator_.labels_)

    def test_reproducibility_with_random_state(self, simple_data):
        """Test that results are reproducible with same random_state."""
        X, _ = simple_data

        bpk1 = BestPolisKMeans(k_bounds=[2, 5], random_state=42)
        bpk1.fit(X)

        bpk2 = BestPolisKMeans(k_bounds=[2, 5], random_state=42)
        bpk2.fit(X)

        assert bpk1.best_k_ == bpk2.best_k_

        # Ensure best_score_ is not None before comparison
        assert bpk1.best_score_ is not None
        assert bpk2.best_score_ is not None
        assert_allclose(bpk1.best_score_, bpk2.best_score_)

        # Access labels through best_estimator_
        assert bpk1.best_estimator_ is not None
        assert bpk2.best_estimator_ is not None
        assert_array_equal(bpk1.best_estimator_.labels_, bpk2.best_estimator_.labels_)

    def test_different_random_states_may_differ(self, simple_data):
        """Test that different random_states may produce different results."""
        X, _ = simple_data

        bpk1 = BestPolisKMeans(k_bounds=[2, 5], random_state=42)
        bpk1.fit(X)

        bpk2 = BestPolisKMeans(k_bounds=[2, 5], random_state=123)
        bpk2.fit(X)

        # Results may differ, but should still be valid
        assert bpk1.best_estimator_ is not None
        assert bpk2.best_estimator_ is not None

    def test_single_k_value(self, simple_data):
        """Test with k_bounds containing a single value."""
        X, _ = simple_data
        bpk = BestPolisKMeans(k_bounds=[3, 3], random_state=42)
        bpk.fit(X)

        assert bpk.best_k_ == 3

    def test_different_init_strategies(self, simple_data):
        """Test that different init strategies work."""
        X, _ = simple_data

        for init_strategy in ["polis", "k-means++", "random"]:
            bpk = BestPolisKMeans(k_bounds=[2, 4], init=init_strategy, random_state=42)
            bpk.fit(X)

            assert bpk.best_k_ is not None
            assert bpk.best_score_ is not None
            assert bpk.best_estimator_ is not None

    def test_with_init_centers(self, simple_data):
        """Test fitting with custom init_centers."""
        X, _ = simple_data
        init_centers = X[:2]  # Use first 2 points as initial centers

        bpk = BestPolisKMeans(
            k_bounds=[2, 4],
            init_centers=init_centers,
            random_state=42
        )
        bpk.fit(X)

        assert bpk.best_k_ is not None
        assert bpk.best_estimator_ is not None

    def test_labels_property(self, simple_data):
        """Test that labels are accessible through best_estimator_ after fitting."""
        X, _ = simple_data
        bpk = BestPolisKMeans(k_bounds=[2, 4], random_state=42)
        bpk.fit(X)

        assert bpk.best_estimator_ is not None
        assert hasattr(bpk.best_estimator_, 'labels_')
        assert len(bpk.best_estimator_.labels_) == len(X)

    def test_cluster_centers_accessible(self, simple_data):
        """Test that cluster centers are accessible through best_estimator_."""
        X, _ = simple_data
        bpk = BestPolisKMeans(k_bounds=[2, 4], random_state=42)
        bpk.fit(X)

        assert bpk.best_estimator_ is not None
        centers = bpk.best_estimator_.cluster_centers_
        assert centers.shape[0] == bpk.best_k_
        assert centers.shape[1] == X.shape[1]

    def test_fit_returns_self(self, simple_data):
        """Test that fit() returns self for method chaining."""
        X, _ = simple_data
        bpk = BestPolisKMeans(k_bounds=[2, 4], random_state=42)
        result = bpk.fit(X)

        assert result is bpk

    def test_minimum_samples(self):
        """Test with minimum number of samples."""
        X = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]])
        bpk = BestPolisKMeans(k_bounds=[2, 3], random_state=42)
        bpk.fit(X)

        assert bpk.best_k_ in [2, 3]
        assert bpk.best_estimator_ is not None
        assert len(bpk.best_estimator_.labels_) == 6