import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from sklearn.datasets import make_blobs
from reddwarf.sklearn.cluster import BestPolisKMeans, PolisKMeans


class TestPolisKMeans:
    """Test suite for PolisKMeans class."""

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
        """Test that PolisKMeans initializes with default parameters."""
        pkm = PolisKMeans()

        assert pkm.n_clusters == 8
        assert pkm._init_strategy == "k-means++"
        assert pkm.init_centers is None
        assert pkm.n_init == "auto"
        assert pkm.max_iter == 300
        assert pkm.tol == 1e-4
        assert pkm.verbose == 0
        assert pkm.random_state is None
        assert pkm.copy_x is True
        assert pkm.algorithm == "lloyd"
        assert pkm.init_centers_used_ is None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        init_centers = np.array([[0, 0], [1, 1], [2, 2]])
        pkm = PolisKMeans(
            n_clusters=3,
            init="polis",
            init_centers=init_centers,
            max_iter=500,
            random_state=42
        )

        assert pkm.n_clusters == 3
        assert pkm._init_strategy == "polis"
        assert pkm.init_centers is not None
        assert_array_equal(pkm.init_centers, init_centers)
        assert pkm.max_iter == 500
        assert pkm.random_state == 42

    def test_fit_sets_init_centers_used(self, simple_data):
        """Test that fit() sets init_centers_used_ attribute."""
        X, _ = simple_data
        pkm = PolisKMeans(n_clusters=3, random_state=42)
        pkm.fit(X)

        assert pkm.init_centers_used_ is not None
        assert pkm.init_centers_used_.shape == (3, 2)

    def test_init_centers_full_specification(self, simple_data):
        """Test with init_centers exactly matching n_clusters."""
        X, _ = simple_data
        init_centers = X[:3]  # 3 centers for 3 clusters

        pkm = PolisKMeans(n_clusters=3, init_centers=init_centers, random_state=42)
        pkm.fit(X)

        assert pkm.init_centers_used_ is not None
        assert_array_equal(pkm.init_centers_used_, init_centers)

    def test_init_centers_fewer_than_n_clusters(self, simple_data):
        """Test with init_centers having fewer centers than n_clusters."""
        X, _ = simple_data
        init_centers = X[:2]  # 2 centers for 3 clusters

        pkm = PolisKMeans(n_clusters=3, init_centers=init_centers, random_state=42)
        pkm.fit(X)

        # First 2 should match provided centers
        # First 2 should match provided centers
        assert pkm.init_centers_used_ is not None
        assert_array_equal(pkm.init_centers_used_[:2], init_centers)
        # Should have 3 total centers
        assert pkm.init_centers_used_.shape == (3, 2)

    def test_init_centers_more_than_n_clusters(self, simple_data):
        """Test with init_centers having more centers than n_clusters."""
        X, _ = simple_data
        init_centers = X[:5]  # 5 centers for 3 clusters

        pkm = PolisKMeans(n_clusters=3, init_centers=init_centers, random_state=42)
        pkm.fit(X)

        # Should be trimmed to first 3
        # Should be trimmed to first 3
        assert pkm.init_centers_used_ is not None
        assert_array_equal(pkm.init_centers_used_, init_centers[:3])
        assert pkm.init_centers_used_.shape == (3, 2)

    def test_init_strategy_kmeans_plusplus(self, simple_data):
        """Test k-means++ initialization strategy."""
        X, _ = simple_data
        pkm = PolisKMeans(n_clusters=3, init="k-means++", random_state=42)
        pkm.fit(X)

        assert pkm.init_centers_used_ is not None
        assert pkm.init_centers_used_.shape == (3, 2)

    def test_init_strategy_random(self, simple_data):
        """Test random initialization strategy."""
        X, _ = simple_data
        pkm = PolisKMeans(n_clusters=3, init="random", random_state=42)
        pkm.fit(X)

        assert pkm.init_centers_used_ is not None
        assert pkm.init_centers_used_.shape == (3, 2)

    def test_init_strategy_polis(self, simple_data):
        """Test polis initialization strategy."""
        X, _ = simple_data
        pkm = PolisKMeans(n_clusters=3, init="polis", random_state=42)
        pkm.fit(X)

        assert pkm.init_centers_used_ is not None
        assert pkm.init_centers_used_.shape == (3, 2)

    def test_polis_strategy_uses_unique_rows(self):
        """Test that polis strategy selects first unique rows."""
        X = np.array([
            [0, 0],
            [1, 1],
            [1, 1],  # duplicate
            [2, 2],
            [3, 3],
        ])

        pkm = PolisKMeans(n_clusters=3, init="polis")
        pkm.fit(X)

        # Should use first 3 unique rows: [0,0], [1,1], [2,2]
        expected = np.array([[0, 0], [1, 1], [2, 2]])
        # Should use first 3 unique rows: [0,0], [1,1], [2,2]
        expected = np.array([[0, 0], [1, 1], [2, 2]])
        assert pkm.init_centers_used_ is not None
        assert_array_equal(pkm.init_centers_used_, expected)

    def test_polis_strategy_insufficient_unique_rows(self):
        """Test that polis strategy raises error with insufficient unique rows."""
        X = np.array([
            [0, 0],
            [0, 0],  # duplicate
            [1, 1],
        ])

        pkm = PolisKMeans(n_clusters=3, init="polis")

        with pytest.raises(ValueError, match="Not enough unique rows"):
            pkm.fit(X)

    def test_invalid_init_centers_shape(self, simple_data):
        """Test that invalid init_centers shape raises error."""
        X, _ = simple_data
        init_centers = np.array([0, 1, 2])  # 1D array

        pkm = PolisKMeans(n_clusters=3, init_centers=init_centers)

        with pytest.raises(ValueError, match="init_centers must be of shape"):
            pkm.fit(X)

    def test_init_centers_wrong_n_features(self, simple_data):
        """Test that init_centers with wrong number of features raises error."""
        X, _ = simple_data  # 2 features
        init_centers = np.array([[0, 1, 2], [3, 4, 5]])  # 3 features

        pkm = PolisKMeans(n_clusters=2, init_centers=init_centers)

        with pytest.raises(ValueError, match="init_centers must be of shape"):
            pkm.fit(X)

    def test_unsupported_init_strategy(self, simple_data):
        """Test that unsupported init strategy raises error."""
        X, _ = simple_data
        pkm = PolisKMeans(n_clusters=3, init="invalid")
        pkm._init_strategy = "invalid"  # Bypass __init__ validation

        with pytest.raises(ValueError, match="Unsupported init strategy"):
            pkm.fit(X)

    def test_reproducibility_with_random_state(self, simple_data):
        """Test that results are reproducible with same random_state."""
        X, _ = simple_data

        pkm1 = PolisKMeans(n_clusters=3, init="k-means++", random_state=42)
        pkm1.fit(X)

        pkm2 = PolisKMeans(n_clusters=3, init="k-means++", random_state=42)
        pkm2.fit(X)

        assert pkm1.init_centers_used_ is not None
        assert pkm2.init_centers_used_ is not None
        assert_array_equal(pkm1.init_centers_used_, pkm2.init_centers_used_)
        assert pkm1.labels_ is not None
        assert pkm2.labels_ is not None
        assert_array_equal(pkm1.labels_, pkm2.labels_)
        assert_allclose(pkm1.cluster_centers_, pkm2.cluster_centers_)

    def test_polis_strategy_deterministic(self, simple_data):
        """Test that polis strategy is deterministic regardless of random_state."""
        X, _ = simple_data

        pkm1 = PolisKMeans(n_clusters=3, init="polis", random_state=42)
        pkm1.fit(X)

        pkm2 = PolisKMeans(n_clusters=3, init="polis", random_state=123)
        pkm2.fit(X)

        # Polis should give same init_centers regardless of random_state
        # Polis should give same init_centers regardless of random_state
        assert pkm1.init_centers_used_ is not None
        assert pkm2.init_centers_used_ is not None
        assert_array_equal(pkm1.init_centers_used_, pkm2.init_centers_used_)

    def test_partial_init_centers_with_kmeans_plusplus(self, simple_data):
        """Test partial init_centers filled with k-means++."""
        X, _ = simple_data
        init_centers = X[:1]  # 1 center for 3 clusters

        pkm = PolisKMeans(
            n_clusters=3,
            init="k-means++",
            init_centers=init_centers,
            random_state=42
        )
        pkm.fit(X)

        # First center should match provided
        # First center should match provided
        assert pkm.init_centers_used_ is not None
        assert_array_equal(pkm.init_centers_used_[0], init_centers[0])
        # Should have 3 total
        assert pkm.init_centers_used_.shape == (3, 2)

    def test_partial_init_centers_with_random(self, simple_data):
        """Test partial init_centers filled with random strategy."""
        X, _ = simple_data
        init_centers = X[:1]  # 1 center for 3 clusters

        pkm = PolisKMeans(
            n_clusters=3,
            init="random",
            init_centers=init_centers,
            random_state=42
        )
        pkm.fit(X)

        # First center should match provided
        # First center should match provided
        assert pkm.init_centers_used_ is not None
        assert_array_equal(pkm.init_centers_used_[0], init_centers[0])
        # Should have 3 total
        assert pkm.init_centers_used_.shape == (3, 2)

    def test_partial_init_centers_with_polis(self, simple_data):
        """Test partial init_centers filled with polis strategy."""
        X, _ = simple_data
        init_centers = X[:1]  # 1 center for 3 clusters

        pkm = PolisKMeans(
            n_clusters=3,
            init="polis",
            init_centers=init_centers,
            random_state=42
        )
        pkm.fit(X)

        # First center should match provided
        # First center should match provided
        assert pkm.init_centers_used_ is not None
        assert_array_equal(pkm.init_centers_used_[0], init_centers[0])
        # Should have 3 total
        assert pkm.init_centers_used_.shape == (3, 2)

    def test_fit_produces_cluster_centers(self, simple_data):
        """Test that fit produces cluster_centers_ attribute."""
        X, _ = simple_data
        pkm = PolisKMeans(n_clusters=3, random_state=42)
        pkm.fit(X)

        assert hasattr(pkm, 'cluster_centers_')
        assert pkm.cluster_centers_.shape == (3, 2)

    def test_fit_produces_labels(self, simple_data):
        """Test that fit produces labels_ attribute."""
        X, _ = simple_data
        pkm = PolisKMeans(n_clusters=3, random_state=42)
        pkm.fit(X)

        assert hasattr(pkm, 'labels_')
        assert hasattr(pkm, 'labels_')
        assert pkm.labels_ is not None
        assert len(pkm.labels_) == len(X)
        assert pkm.labels_.min() == 0
        assert pkm.labels_.max() <= 2

    def test_fit_predict(self, simple_data):
        """Test fit_predict method."""
        X, _ = simple_data
        pkm = PolisKMeans(n_clusters=3, random_state=42)
        labels = pkm.fit_predict(X)

        assert len(labels) == len(X)
        assert labels.min() == 0
        assert labels.max() <= 2

    def test_predict_after_fit(self, simple_data):
        """Test predict method after fitting."""
        X, _ = simple_data
        pkm = PolisKMeans(n_clusters=3, random_state=42)
        pkm.fit(X)

        # Predict on same data
        labels = pkm.predict(X)
        # Predict on same data
        labels = pkm.predict(X)
        assert pkm.labels_ is not None
        assert_array_equal(labels, pkm.labels_)

    def test_init_centers_not_modified(self, simple_data):
        """Test that provided init_centers are not modified in place."""
        X, _ = simple_data
        init_centers = X[:2].copy()
        original = init_centers.copy()

        pkm = PolisKMeans(n_clusters=3, init_centers=init_centers, random_state=42)
        pkm.fit(X)

        # Original init_centers should not be modified
        assert_array_equal(init_centers, original)

    def test_higher_dimensional_data(self):
        """Test with higher dimensional data."""
        X, _ = make_blobs(n_samples=150, centers=4, n_features=10, random_state=42)  # type: ignore
        pkm = PolisKMeans(n_clusters=4, random_state=42)
        pkm.fit(X)

        assert pkm.cluster_centers_.shape == (4, 10)
        assert pkm.cluster_centers_.shape == (4, 10)
        assert pkm.init_centers_used_ is not None
        assert pkm.init_centers_used_.shape == (4, 10)

    def test_fit_with_sample_weight(self, simple_data):
        """Test fitting with sample weights."""
        X, _ = simple_data
        sample_weight = np.random.rand(len(X))

        pkm = PolisKMeans(n_clusters=3, random_state=42)
        pkm.fit(X, sample_weight=sample_weight)

        assert hasattr(pkm, 'cluster_centers_')
        assert hasattr(pkm, 'labels_')

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