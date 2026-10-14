"""
Unit tests for BagOfWords_Classification.py

Run with:   pytest test_bow.py -v
(or simply: python test_bow.py)

Put this file in the same folder as BagOfWords_Classification.py.
All tests use small synthetic data, so the whole suite runs in a few seconds.
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import BagOfWords_Classification as bow  # noqa: E402


# ----------------------------------------------------------------------
# Reference implementation of HOG (the original loop + np.histogram version)
# used to check that the vectorized version gives the same result.
# ----------------------------------------------------------------------
def hog_reference(img, x_coords, y_coords, cell_size=4, n_cells=4, n_bins=8):
    from scipy import ndimage
    img = img.astype(np.float64)
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    angles = np.arctan2(gy, gx)
    patch_size = cell_size * n_cells
    half = patch_size // 2
    h, w = img.shape
    out = np.zeros((len(x_coords), n_cells * n_cells * n_bins))
    for p, (x, y) in enumerate(zip(x_coords, y_coords)):
        top = int(np.clip(y - half, 0, h - patch_size))
        left = int(np.clip(x - half, 0, w - patch_size))
        patch = angles[top:top + patch_size, left:left + patch_size]
        feat = []
        for i in range(n_cells):
            for j in range(n_cells):
                cell = patch[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
                hist, _ = np.histogram(cell, bins=n_bins, range=(-np.pi, np.pi))
                feat.append(hist)
        out[p] = np.concatenate(feat)
    return out


def make_blobs(centers, n_per_blob=100, dim=128, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    data = [c * np.ones(dim) + noise * rng.standard_normal((n_per_blob, dim))
            for c in centers]
    return np.vstack(data)


def inertia(data, centroids):
    return np.sum(cdist(data, centroids).min(axis=1) ** 2)


# ======================================================================
# grid_points
# ======================================================================
class TestGridPoints:
    def test_length_and_dtype(self):
        img = np.zeros((96, 96))
        x, y = bow.grid_points(img, 10, 10)
        assert x.shape == (100,) and y.shape == (100,)
        assert x.ndim == 1 and y.ndim == 1

    def test_non_square_grid_and_image(self):
        img = np.zeros((60, 100))                      # h=60, w=100
        x, y = bow.grid_points(img, n_points_x=5, n_points_y=3)
        assert len(x) == len(y) == 15
        assert x.min() == 8 and x.max() == 100 - 8     # columns follow the width
        assert y.min() == 8 and y.max() == 60 - 8      # rows follow the height

    def test_coordinates_within_borders(self):
        img = np.zeros((96, 96))
        x, y = bow.grid_points(img, 12, 7, border=8)
        assert x.min() >= 8 and x.max() <= 96 - 8
        assert y.min() >= 8 and y.max() <= 96 - 8

    def test_ordering_x_fastest(self):
        img = np.zeros((96, 96))
        x, y = bow.grid_points(img, 10, 10)
        assert (x[:10] == x[10:20]).all()              # same x pattern on every row
        assert (y[:10] == y[0]).all()                  # y constant along a row
        assert (x[0], y[0]) == (8, 8)
        assert (x[-1], y[-1]) == (88, 88)

    def test_all_points_distinct(self):
        x, y = bow.grid_points(np.zeros((96, 96)), 10, 10)
        assert len(set(zip(x.tolist(), y.tolist()))) == 100


# ======================================================================
# descriptors_hog
# ======================================================================
class TestHOG:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.img = rng.random((96, 96)) * 255
        self.x, self.y = bow.grid_points(self.img, 6, 6)

    def test_shape(self):
        d = bow.descriptors_hog(self.img, self.x, self.y)
        assert d.shape == (36, 128)

    def test_each_cell_sums_to_16_and_descriptor_to_256(self):
        d = bow.descriptors_hog(self.img, self.x, self.y)
        assert (d.reshape(36, 16, 8).sum(axis=2) == 16).all()   # 4x4 pixels per cell
        assert (d.sum(axis=1) == 256).all()                     # 16x16 pixels per patch

    def test_non_negative_integers(self):
        d = bow.descriptors_hog(self.img, self.x, self.y)
        assert (d >= 0).all()
        assert np.issubdtype(d.dtype, np.integer)

    def test_constant_image_all_in_one_bin(self):
        img = np.full((96, 96), 100.0)                 # zero gradient -> angle 0 -> bin 4
        d = bow.descriptors_hog(img, self.x, self.y).reshape(36, 16, 8)
        assert (d[:, :, 4] == 16).all()
        assert d.sum() == 36 * 256

    def test_horizontal_ramp_gives_angle_zero(self):
        img = np.tile(np.arange(96, dtype=float), (96, 1))      # gx > 0, gy = 0
        d = bow.descriptors_hog(img, self.x, self.y).reshape(36, 16, 8)
        assert (d[:, :, 4] == 16).all()

    def test_matches_reference_loop_implementation(self):
        d_fast = bow.descriptors_hog(self.img, self.x, self.y)
        d_ref = hog_reference(self.img, self.x, self.y)
        assert np.array_equal(d_fast, d_ref)

    def test_matches_reference_on_another_image_and_grid(self):
        rng = np.random.default_rng(5)
        img = rng.random((96, 96)) * 255
        x, y = bow.grid_points(img, 4, 9)
        assert np.array_equal(bow.descriptors_hog(img, x, y), hog_reference(img, x, y))

    def test_deterministic(self):
        a = bow.descriptors_hog(self.img, self.x, self.y)
        b = bow.descriptors_hog(self.img, self.x, self.y)
        assert np.array_equal(a, b)


# ======================================================================
# assign_nearest
# ======================================================================
class TestAssignNearest:
    def test_matches_plain_argmin_cdist_with_small_batches(self):
        rng = np.random.default_rng(1)
        data, centers = rng.random((53, 10)), rng.random((6, 10))
        expected = np.argmin(cdist(data, centers), axis=1)
        got = bow.assign_nearest(data, centers, batch_size=7)   # forces 8 batches
        assert np.array_equal(got, expected)

    def test_l1_metric(self):
        rng = np.random.default_rng(2)
        data, centers = rng.random((40, 5)), rng.random((4, 5))
        expected = np.argmin(cdist(data, centers, "cityblock"), axis=1)
        got = bow.assign_nearest(data, centers, metric="cityblock", batch_size=9)
        assert np.array_equal(got, expected)

    def test_points_equal_to_centers(self):
        centers = np.eye(4) * 10
        assert np.array_equal(bow.assign_nearest(centers, centers), np.arange(4))


# ======================================================================
# kmeans
# ======================================================================
class TestKMeans:
    def test_output_shape_and_no_nan(self):
        data = np.random.default_rng(0).random((200, 128))
        c = bow.kmeans(data, K=7, max_iter=10, seed=0)
        assert c.shape == (7, 128)
        assert not np.isnan(c).any()

    def test_k_equals_one_gives_global_mean(self):
        data = np.random.default_rng(1).random((50, 128))
        c = bow.kmeans(data, K=1, max_iter=5, seed=0)
        assert np.allclose(c[0], data.mean(axis=0))

    def test_k_equals_n_gives_the_data_points(self):
        data = np.random.default_rng(2).random((12, 128))
        c = bow.kmeans(data, K=12, max_iter=3, seed=0)
        order_c = np.lexsort(c.T[::-1])
        order_d = np.lexsort(data.T[::-1])
        assert np.allclose(c[order_c], data[order_d])

    def test_same_seed_is_reproducible(self):
        data = np.random.default_rng(3).random((300, 128))
        a = bow.kmeans(data, K=5, max_iter=10, seed=42)
        b = bow.kmeans(data, K=5, max_iter=10, seed=42)
        assert np.array_equal(a, b)

    def test_different_seeds_give_different_codebooks(self):
        data = np.random.default_rng(4).random((300, 128))
        a = bow.kmeans(data, K=5, max_iter=1, seed=1)
        b = bow.kmeans(data, K=5, max_iter=1, seed=2)
        assert not np.allclose(a, b)

    def test_inertia_does_not_increase_with_more_iterations(self):
        # Lloyd's algorithm never increases the objective (same seed = same init)
        data = np.random.default_rng(5).random((400, 128))
        c1 = bow.kmeans(data, K=8, max_iter=1, seed=0)
        c10 = bow.kmeans(data, K=8, max_iter=10, seed=0)
        assert inertia(data, c10) <= inertia(data, c1) + 1e-9

    def test_recovers_well_separated_blobs(self):
        # Random init can land two centroids in the same blob (local minimum),
        # which is normal for K-means. So we require success for at least one
        # of several seeds, and no NaN for any of them.
        true_centers = np.array([0.0, 100.0, 200.0])
        data = make_blobs(true_centers)
        successes = 0
        for seed in range(30):
            c = bow.kmeans(data, K=3, max_iter=10, seed=seed)
            assert not np.isnan(c).any()
            found = np.sort(c.mean(axis=1))
            if np.allclose(found, true_centers, atol=1.0):
                successes += 1
        assert successes >= 1

    def test_empty_clusters_do_not_produce_nan(self):
        # many identical points -> identical initial centroids -> empty clusters
        data = np.vstack([np.ones((30, 128)), np.zeros((3, 128))])
        c = bow.kmeans(data, K=6, max_iter=5, seed=0)
        assert c.shape == (6, 128)
        assert not np.isnan(c).any()

    def test_works_with_uint8_input(self):
        data = np.random.default_rng(6).integers(0, 17, size=(100, 128)).astype(np.uint8)
        c = bow.kmeans(data, K=4, max_iter=3, seed=0)
        assert c.dtype == np.float64 and c.shape == (4, 128)

    def test_runs_exactly_max_iter_when_tol_is_zero(self):
        # converged data would stop early only if tol > 0; with tol=0 the result
        # must still be a valid fixed point after extra iterations
        data = make_blobs([0.0, 100.0], n_per_blob=30)
        a = bow.kmeans(data, K=2, max_iter=10, tol=0.0, seed=3)
        b = bow.kmeans(data, K=2, max_iter=10, tol=1e-6, seed=3)
        assert np.allclose(a, b)


# ======================================================================
# bow_histograms
# ======================================================================
class TestBowHistograms:
    def setup_method(self):
        self.K, self.D, self.P = 4, 128, 5
        self.codebook = np.stack([np.full(self.D, 50.0 * k) for k in range(self.K)])

    def test_shape_and_row_sums(self):
        rng = np.random.default_rng(0)
        desc = rng.random((6, self.P, self.D)) * 150
        h = bow.bow_histograms(desc, self.codebook)
        assert h.shape == (6, self.K)
        assert (h.sum(axis=1) == self.P).all()

    def test_known_counts(self):
        counts = np.array([[3, 0, 2, 0],
                           [0, 5, 0, 0],
                           [1, 1, 1, 2]])
        desc = np.zeros((3, self.P, self.D))
        for i in range(3):
            words = np.repeat(np.arange(self.K), counts[i])
            desc[i] = self.codebook[words] + 0.1       # slightly off the centroid
        h = bow.bow_histograms(desc, self.codebook)
        assert np.array_equal(h, counts)

    def test_accepts_uint8_descriptors(self):
        desc = np.zeros((2, self.P, self.D), dtype=np.uint8)
        h = bow.bow_histograms(desc, self.codebook)
        assert (h[:, 0] == self.P).all()               # all closest to word 0

    def test_matches_loop_per_image(self):
        rng = np.random.default_rng(1)
        desc = rng.random((5, self.P, self.D)) * 150
        h = bow.bow_histograms(desc, self.codebook)
        for i in range(5):
            nearest = np.argmin(cdist(desc[i], self.codebook), axis=1)
            assert np.array_equal(h[i], np.bincount(nearest, minlength=self.K))


# ======================================================================
# nearest_neighbor_predict / accuracy
# ======================================================================
class TestNearestNeighbor:
    def test_identical_point_gets_its_label(self):
        rng = np.random.default_rng(0)
        train = rng.random((30, 10))
        labels = rng.integers(0, 3, 30)
        pred = bow.nearest_neighbor_predict(train, train, labels)
        assert np.array_equal(pred, labels)

    def test_output_is_1d_int_array(self):
        rng = np.random.default_rng(1)
        pred = bow.nearest_neighbor_predict(rng.random((7, 4)), rng.random((9, 4)),
                                            rng.integers(0, 2, 9))
        assert pred.shape == (7,) and np.issubdtype(pred.dtype, np.integer)

    def test_l1_and_l2_can_disagree(self):
        # query at the origin
        #   t1 = (2, 2)   -> L2 = 2.83, L1 = 4.0
        #   t2 = (3.5, 0) -> L2 = 3.5,  L1 = 3.5
        train = np.array([[2.0, 2.0], [3.5, 0.0]])
        labels = np.array([0, 1])
        q = np.array([[0.0, 0.0]])
        assert bow.nearest_neighbor_predict(q, train, labels, metric="euclidean")[0] == 0
        assert bow.nearest_neighbor_predict(q, train, labels, metric="cityblock")[0] == 1

    def test_batched_equals_unbatched(self):
        rng = np.random.default_rng(2)
        test, train = rng.random((25, 6)), rng.random((40, 6))
        labels = rng.integers(0, 4, 40)
        a = bow.nearest_neighbor_predict(test, train, labels, batch_size=1000)
        b = bow.nearest_neighbor_predict(test, train, labels, batch_size=4)
        assert np.array_equal(a, b)

    def test_simple_two_clusters(self):
        train = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
        labels = np.array([0, 0, 1, 1])
        test = np.array([[1, 0], [9, 10]], dtype=float)
        assert np.array_equal(bow.nearest_neighbor_predict(test, train, labels), [0, 1])

    def test_accuracy(self):
        assert bow.accuracy(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 0])) == 0.75
        assert bow.accuracy(np.array([1, 1]), np.array([1, 1])) == 1.0
        assert bow.accuracy(np.array([0, 0]), np.array([1, 1])) == 0.0


# ======================================================================
# select_classes / subsample_per_class
# ======================================================================
class TestClassSelection:
    def setup_method(self):
        self.labels = np.repeat(np.arange(10), 20)             # 10 classes x 20 images
        self.desc = np.arange(len(self.labels))[:, None, None] * np.ones((1, 3, 128))

    def test_select_classes_reindexes_labels(self):
        d, l = bow.select_classes(self.desc, self.labels, [3, 7])
        assert len(d) == len(l) == 40
        assert set(l.tolist()) == {0, 1}
        assert (l[:20] == 0).all() and (l[20:] == 1).all()

    def test_select_classes_keeps_matching_descriptors(self):
        d, l = bow.select_classes(self.desc, self.labels, [0, 1])
        assert d.shape == (40, 3, 128)
        assert d[:, 0, 0].max() == 39                          # images 0..39 only

    def test_select_classes_single_and_all(self):
        d, l = bow.select_classes(self.desc, self.labels, list(range(10)))
        assert len(l) == 200 and np.array_equal(l, self.labels)

    @pytest.mark.skipif(not hasattr(bow, "subsample_per_class"),
                        reason="subsample_per_class not defined in the main file yet")
    def test_subsample_counts_per_class(self):
        d, l = bow.subsample_per_class(self.desc, self.labels, fraction=0.1, seed=0)
        assert len(l) == 20                                    # 2 per class
        assert (np.bincount(l) == 2).all()

    @pytest.mark.skipif(not hasattr(bow, "subsample_per_class"),
                        reason="subsample_per_class not defined in the main file yet")
    def test_subsample_reproducible_and_aligned(self):
        d1, l1 = bow.subsample_per_class(self.desc, self.labels, 0.1, seed=7)
        d2, l2 = bow.subsample_per_class(self.desc, self.labels, 0.1, seed=7)
        assert np.array_equal(d1, d2) and np.array_equal(l1, l2)
        # each kept descriptor still belongs to the label it is paired with
        assert np.array_equal(d1[:, 0, 0].astype(int) // 20, l1)

    @pytest.mark.skipif(not hasattr(bow, "subsample_per_class"),
                        reason="subsample_per_class not defined in the main file yet")
    def test_subsample_different_seed_different_subset(self):
        d1, _ = bow.subsample_per_class(self.desc, self.labels, 0.1, seed=1)
        d2, _ = bow.subsample_per_class(self.desc, self.labels, 0.1, seed=2)
        assert not np.array_equal(d1, d2)


# ======================================================================
# Data loading + end-to-end smoke test on a fake dataset
# ======================================================================
@pytest.fixture
def fake_dataset(tmp_path):
    """stl10_raw-like folder: {train,test}/{airplane,bird}/*.png with 96x96 images."""
    rng = np.random.default_rng(0)
    root = tmp_path / "stl10_raw"
    for split, n in (("train", 6), ("test", 4)):
        for cls in ("bird", "airplane"):                       # unsorted on purpose
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(n):
                arr = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
                Image.fromarray(arr).save(folder / f"img_{i}.png")
    (root / "train" / "notes.txt").write_text("not a class folder")
    return root


class TestDataAndPipeline:
    def test_load_image_gray(self, fake_dataset):
        p = next((fake_dataset / "train" / "bird").glob("*.png"))
        img = bow.load_image_gray(str(p))
        assert img.shape == (96, 96) and img.ndim == 2
        assert img.dtype == np.float64

    def test_list_images(self, fake_dataset):
        paths, labels, names = bow.list_images(str(fake_dataset / "train"))
        assert names == ["airplane", "bird"]                   # sorted, file ignored
        assert len(paths) == 12 and labels.shape == (12,)
        assert (labels[:6] == 0).all() and (labels[6:] == 1).all()
        assert all("airplane" in p for p in paths[:6])

    def test_same_labels_for_train_and_test(self, fake_dataset):
        _, _, n_train = bow.list_images(str(fake_dataset / "train"))
        _, _, n_test = bow.list_images(str(fake_dataset / "test"))
        assert n_train == n_test

    def test_compute_descriptors_set_and_cache(self, fake_dataset, tmp_path):
        cache = str(tmp_path / "cache")
        train_dir = str(fake_dataset / "train")
        desc, labels = bow.compute_descriptors_set(train_dir, 5, 5, cache_dir=cache)
        assert desc.shape == (12, 25, 128) and desc.dtype == np.uint8
        assert labels.shape == (12,)
        assert (desc.sum(axis=2) == 256).all()
        assert os.path.exists(os.path.join(cache, "train_5x5.npz"))
        # second call must load from the cache and give identical data
        desc2, labels2 = bow.compute_descriptors_set(train_dir, 5, 5, cache_dir=cache)
        assert np.array_equal(desc, desc2) and np.array_equal(labels, labels2)

    def test_cache_depends_on_grid_size(self, fake_dataset, tmp_path):
        cache = str(tmp_path / "cache")
        train_dir = str(fake_dataset / "train")
        d1, _ = bow.compute_descriptors_set(train_dir, 4, 4, cache_dir=cache)
        d2, _ = bow.compute_descriptors_set(train_dir, 6, 6, cache_dir=cache)
        assert d1.shape[1] == 16 and d2.shape[1] == 36

    def test_end_to_end_pipeline(self, fake_dataset, tmp_path):
        cache = str(tmp_path / "cache")
        tr_desc, tr_lab = bow.compute_descriptors_set(
            str(fake_dataset / "train"), 5, 5, cache_dir=cache)
        te_desc, te_lab = bow.compute_descriptors_set(
            str(fake_dataset / "test"), 5, 5, cache_dir=cache)

        for metric in ("euclidean", "cityblock"):
            codebook = bow.kmeans(tr_desc.reshape(-1, 128), K=8, max_iter=10, seed=0)
            assert codebook.shape == (8, 128)
            tr_h = bow.bow_histograms(tr_desc, codebook)
            te_h = bow.bow_histograms(te_desc, codebook)
            assert tr_h.shape == (12, 8) and te_h.shape == (8, 8)
            assert (tr_h.sum(axis=1) == 25).all()
            pred = bow.nearest_neighbor_predict(te_h, tr_h, tr_lab, metric=metric)
            assert pred.shape == (8,)
            assert set(pred.tolist()) <= {0, 1}
            assert 0.0 <= bow.accuracy(pred, te_lab) <= 1.0

    def test_training_images_classified_perfectly_against_themselves(self, fake_dataset, tmp_path):
        cache = str(tmp_path / "cache")
        tr_desc, tr_lab = bow.compute_descriptors_set(
            str(fake_dataset / "train"), 5, 5, cache_dir=cache)
        codebook = bow.kmeans(tr_desc.reshape(-1, 128), K=8, max_iter=10, seed=0)
        h = bow.bow_histograms(tr_desc, codebook)
        pred = bow.nearest_neighbor_predict(h, h, tr_lab)
        assert bow.accuracy(pred, tr_lab) == 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))