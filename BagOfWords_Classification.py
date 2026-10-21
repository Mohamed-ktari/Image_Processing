import os
import glob
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import cdist


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_image_gray(path):
    """Load an image as a 2D float grayscale array."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def list_images(split_dir):
    """
    List all image paths of a split with integer labels (from sorted class folder names).
    Returns paths (list), labels (1D int array), class_names (sorted list)
    """
    class_names = sorted(d for d in os.listdir(split_dir)
                         if os.path.isdir(os.path.join(split_dir, d)))
    paths, labels = [], []
    for c, name in enumerate(class_names):
        files = sorted(glob.glob(os.path.join(split_dir, name, "*")))
        files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        paths.extend(files)
        labels.extend([c] * len(files))
    return paths, np.array(labels, dtype=int), class_names


def select_classes(desc, labels, class_ids):
    """Keep only the given classes and re-index their labels to 0..C-1."""
    mask = np.isin(labels, class_ids)
    new_labels = np.searchsorted(np.sort(class_ids), labels[mask])
    return desc[mask], new_labels


# ----------------------------------------------------------------------
# Step 1 & 2: grid + HOG
# ----------------------------------------------------------------------
def grid_points(img, n_points_x, n_points_y, border=8):
    """Dense regular grid; returns x (column) and y (row) indices as 1D arrays."""
    h, w = img.shape[:2]
    xs = np.linspace(border, w - border, n_points_x).astype(int)
    ys = np.linspace(border, h - border, n_points_y).astype(int)
    x_grid, y_grid = np.meshgrid(xs, ys)
    return x_grid.ravel(), y_grid.ravel()


def descriptors_hog(img, x_coords, y_coords, cell_size=4, n_cells=4, n_bins=8):
    """
    Vectorized HOG: for each grid point, 16x16 patch -> 4x4 cells -> 8-bin histogram
    of gradient directions per cell -> 128-D descriptor. Output: (n_points, 128).
    """
    img = img.astype(np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)

    gx = ndimage.sobel(img, axis=1)                               # horizontal gradient
    gy = ndimage.sobel(img, axis=0)                               # vertical gradient
    angles = np.arctan2(gy, gx)                                   # [-pi, pi]

    # orientation bin of every pixel, computed once
    bin_map = np.minimum(((angles + np.pi) / (2 * np.pi) * n_bins).astype(int), n_bins - 1)

    patch_size = cell_size * n_cells
    half = patch_size // 2
    h, w = img.shape
    P = len(x_coords)
    D = n_cells * n_cells * n_bins

    tops = np.clip(y_coords - half, 0, h - patch_size)
    lefts = np.clip(x_coords - half, 0, w - patch_size)
    r = np.arange(patch_size)

    # gather all patches at once: (P, 16, 16)
    patches = bin_map[tops[:, None, None] + r[None, :, None],
                      lefts[:, None, None] + r[None, None, :]]

    # cell index of each pixel inside a patch (row-major over the 4x4 cells)
    cell_id = (r[:, None] // cell_size) * n_cells + (r[None, :] // cell_size)

    # one code per pixel = cell * n_bins + bin, offset by point index -> single bincount
    codes = cell_id[None] * n_bins + patches
    codes = codes.reshape(P, -1) + (np.arange(P) * D)[:, None]
    return np.bincount(codes.ravel(), minlength=P * D).reshape(P, D)


# ----------------------------------------------------------------------
# Descriptor extraction with disk cache (done ONCE for all experiments)
# ----------------------------------------------------------------------
def compute_descriptors_set(split_dir, n_points_x, n_points_y, cache_dir="cache"):
    """
    HOG descriptors of every image of a split.
    Returns desc (N_images, n_points, 128) as uint8, labels (N_images,).
    """
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(split_dir))
    cache_file = os.path.join(cache_dir, f"{name}_{n_points_x}x{n_points_y}.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["desc"], data["labels"]

    paths, labels, _ = list_images(split_dir)
    desc = np.zeros((len(paths), n_points_x * n_points_y, 128), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = load_image_gray(p)
        x, y = grid_points(img, n_points_x, n_points_y)
        desc[i] = descriptors_hog(img, x, y)
        if (i + 1) % 1000 == 0:
            print(f"  [{name}] {i + 1}/{len(paths)} images processed")

    np.savez(cache_file, desc=desc, labels=labels)
    return desc, labels


# ----------------------------------------------------------------------
# Step: K-means (always Euclidean, as in the statement)
# ----------------------------------------------------------------------
def assign_nearest(data, centers, metric="euclidean", batch_size=20000):
    """Index of the nearest center for every row of data, computed in batches."""
    out = np.empty(len(data), dtype=int)
    for s in range(0, len(data), batch_size):
        out[s:s + batch_size] = np.argmin(
            cdist(data[s:s + batch_size], centers, metric=metric), axis=1)
    return out


def kmeans(data, K, max_iter=10, tol=0.0, seed=None):
    """
    K-means. data: (N, 128). Returns centroids (K, 128).
    With tol=0 it always runs exactly max_iter iterations.
    """
    rng = np.random.default_rng(seed)
    data = data.astype(np.float64)
    N, D = data.shape

    # Step 1: random initialization with K distinct data points
    centroids = data[rng.permutation(N)[:K]].copy()

    for _ in range(max_iter):
        # Step 2: assignment
        labels = assign_nearest(data, centroids)

        # Step 3: update (per-dimension bincount = fast cluster sums)
        counts = np.bincount(labels, minlength=K)
        new_centroids = np.empty_like(centroids)
        for d in range(D):
            new_centroids[:, d] = np.bincount(labels, weights=data[:, d], minlength=K)
        nonempty = counts > 0
        new_centroids[nonempty] /= counts[nonempty, None]
        # empty clusters: re-seed with random data points
        n_empty = int((~nonempty).sum())
        if n_empty > 0:
            new_centroids[~nonempty] = data[rng.integers(0, N, size=n_empty)]

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift <= tol and tol > 0:
            break

    return centroids


# ----------------------------------------------------------------------
# Step 3: BoW histograms (vectorized over all images)
# ----------------------------------------------------------------------
def bow_histograms(desc, codebook):
    """
    desc: (N_images, n_points, 128); codebook: (K, 128)
    Returns histograms (N_images, K); bin k = nb of descriptors closest to word k.
    """
    N, P, D = desc.shape
    K = codebook.shape[0]
    nearest = assign_nearest(desc.reshape(-1, D), codebook).reshape(N, P)
    codes = nearest + (np.arange(N) * K)[:, None]
    return np.bincount(codes.ravel(), minlength=N * K).reshape(N, K).astype(np.float64)


# ----------------------------------------------------------------------
# Step 4: nearest neighbor classifier
# ----------------------------------------------------------------------
def chi2_cdist(A, B, eps=1e-10, max_elems=8_000_000):
    """
    Chi-square distance between every row of A and every row of B (lecture slide 46):
        d(h, g) = 1/2 * sum_k (h_k - g_k)^2 / (h_k + g_k + eps)
    Bins where both histograms are 0 contribute 0 (eps only avoids the 0/0).
    Computed in chunks of rows of A so the 3-D intermediate array stays small.
    Returns an (len(A), len(B)) matrix.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    out = np.empty((len(A), len(B)))
    rows = max(1, max_elems // (len(B) * A.shape[1]))
    for s in range(0, len(A), rows):
        a = A[s:s + rows][:, None, :]  # (r, 1, K)
        diff = a - B[None, :, :]  # (r, N_train, K)
        out[s:s + rows] = 0.5 * np.sum(diff * diff / (a + B[None, :, :] + eps), axis=2)
    return out


def nearest_neighbor_predict(test_hist, train_hist, train_labels,
                             metric="euclidean", batch_size=1000):
    """
    1-NN classifier.
    metric: 'euclidean' (L2), 'cityblock' (L1) or 'chi2' (chi-square distance).
    Returns predicted labels, 1D int array (N_test,).
    """
    predictions = np.zeros(len(test_hist), dtype=int)
    for s in range(0, len(test_hist), batch_size):
        batch = test_hist[s:s + batch_size]
        if metric == "chi2":
            dists = chi2_cdist(batch, train_hist)
        else:
            dists = cdist(batch, train_hist, metric=metric)
        predictions[s:s + batch_size] = train_labels[np.argmin(dists, axis=1)]
    return predictions


def accuracy(predictions, ground_truth):
    return np.mean(predictions == ground_truth)


def accuracy(predictions, ground_truth):
    return np.mean(predictions == ground_truth)


def subsample_per_class(desc, labels, fraction=0.1, seed=0):
    """Keep a random `fraction` of the images of every class."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = max(1, int(round(fraction * len(idx))))
        keep.append(rng.choice(idx, size=n, replace=False))
    keep = np.sort(np.concatenate(keep))
    return desc[keep], labels[keep]


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def run_experiment(train_desc, train_labels, test_desc, test_labels, K,
                   metrics=("euclidean",), n_runs=10, n_iter=10, base_seed=0):
    """
    Repeat n_runs times: K-means (new random init, n_iter iterations) -> BoW histograms
    -> 1-NN -> accuracy on the test set.

    K-means always uses the Euclidean distance. `metrics` only changes the distance used
    by the nearest-neighbor classifier, and all metrics share the SAME codebook in a given
    run (same seed), so the L1 / L2 comparison is paired and K-means runs only once.

    Returns a dict {metric: array of n_runs accuracies}.
    """
    accs = {m: [] for m in metrics}
    train_flat = train_desc.reshape(-1, train_desc.shape[-1])
    for run in range(n_runs):
        codebook = kmeans(train_flat, K, max_iter=n_iter, tol=0.0, seed=base_seed + run)
        train_hist = bow_histograms(train_desc, codebook)
        test_hist = bow_histograms(test_desc, codebook)
        for m in metrics:
            pred = nearest_neighbor_predict(test_hist, train_hist, train_labels, metric=m)
            accs[m].append(accuracy(pred, test_labels))
    return {m: np.array(a) for m, a in accs.items()}


def random_classifier(n_test, n_classes, rng):
    """Predict a uniformly random class for each test image."""
    return rng.integers(0, n_classes, size=n_test)


def run_random_baseline(test_labels, n_classes, n_runs=10, base_seed=0):
    """Accuracies of the random classifier over n_runs runs."""
    accs = []
    for run in range(n_runs):
        rng = np.random.default_rng(1000 + base_seed + run)
        pred = random_classifier(len(test_labels), n_classes, rng)
        accs.append(accuracy(pred, test_labels))
    return np.array(accs)


def fmt(a):
    return f"{a.mean() * 100:5.2f} +/- {a.std() * 100:4.2f} %"


# ----------------------------------------------------------------------
# Improvement: visual TF-IDF weighting (lecture slides 47-48)
# ----------------------------------------------------------------------
def compute_idf(train_hist):
    """
    Inverse document frequency of every visual word, from the TRAINING histograms only:
        idf_k = log(N / df_k),  df_k = number of training images containing word k.
    A word present in every image gets idf = 0 (not discriminative); a word that appears
    in no training image gets df = 1 to avoid a division by zero.
    """
    N = train_hist.shape[0]
    df = (train_hist > 0).sum(axis=0)
    return np.log(N / np.maximum(df, 1))


def apply_tfidf(hist, idf):
    """tf-idf_ik = tf_ik * idf_k, where tf_ik is the raw count of word k in image i."""
    return hist * idf[None, :]


def run_experiment_tfidf(train_desc, train_labels, test_desc, test_labels, K,
                         metrics=("euclidean",), n_runs=10, n_iter=10, base_seed=0):
    """
    Same protocol as run_experiment, but in each run the SAME codebook is used to
    evaluate both the plain BoW histograms and the tf-idf weighted ones (paired
    comparison). The seeds are the same as in run_experiment, so the "baseline"
    results are identical to the ones obtained before.

    Returns a dict {(variant, metric): array of n_runs accuracies},
    with variant in {"baseline", "tfidf"}.
    """
    accs = {(v, m): [] for v in ("baseline", "tfidf") for m in metrics}
    train_flat = train_desc.reshape(-1, train_desc.shape[-1])
    for run in range(n_runs):
        codebook = kmeans(train_flat, K, max_iter=n_iter, tol=0.0, seed=base_seed + run)
        train_hist = bow_histograms(train_desc, codebook)
        test_hist = bow_histograms(test_desc, codebook)

        idf = compute_idf(train_hist)  # training set only
        train_w = apply_tfidf(train_hist, idf)
        test_w = apply_tfidf(test_hist, idf)

        for m in metrics:
            for variant, (tr, te) in (("baseline", (train_hist, test_hist)),
                                      ("tfidf", (train_w, test_w))):
                pred = nearest_neighbor_predict(te, tr, train_labels, metric=m)
                accs[(variant, m)].append(accuracy(pred, test_labels))
    return {k: np.array(v) for k, v in accs.items()}



# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_results(results, baselines, k_list, save_path="results_vs_K.png", show=True):
    """
    Experiment 3 figure: one subplot per setting (2 classes / 10 classes),
    accuracy (mean +/- std over the runs) as a function of K, for L2 and L1,
    with the random classifier as a dashed horizontal line.

    results   : dict {(setting_name, K): {metric: array of accuracies}}
    baselines : dict {setting_name: array of random-classifier accuracies}
    """
    import matplotlib.pyplot as plt

    names = list(baselines.keys())
    styles = {"euclidean": ("L2", "o-", "tab:blue"), "cityblock": ("L1", "s-", "tab:orange")}
    x = np.arange(len(k_list))

    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        top = 0
        for metric, (label, fmt_, color) in styles.items():
            means = np.array([results[(name, K)][metric].mean() for K in k_list]) * 100
            stds = np.array([results[(name, K)][metric].std() for K in k_list]) * 100
            ax.errorbar(x, means, yerr=stds, fmt=fmt_, color=color, capsize=4, label=label)
            top = max(top, (means + stds).max())
        base = baselines[name].mean() * 100
        ax.axhline(base, color="gray", linestyle="--", label=f"random classifier ({base:.1f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(k_list)
        ax.set_xlabel("K (number of visual words)")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(name)
        ax.set_ylim(max(0, base - 5), top + 5)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("BoW + 1-NN: L1 vs L2 (mean +/- std over runs)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_model_vs_random(results, baselines, K=100, save_path="model_vs_random.png", show=True):
    """
    Experiments 1 & 2 figure: BoW model (L2, given K) vs random classifier,
    for each setting, as bars with std error bars.
    """
    import matplotlib.pyplot as plt

    names = list(baselines.keys())
    model = [results[(n, K)]["euclidean"] for n in names]
    rand = [baselines[n] for n in names]
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.8))
    b1 = ax.bar(x - w / 2, [m.mean() * 100 for m in model], w,
                yerr=[m.std() * 100 for m in model], capsize=4, label=f"BoW model (K={K}, L2)")
    b2 = ax.bar(x + w / 2, [r.mean() * 100 for r in rand], w,
                yerr=[r.std() * 100 for r in rand], capsize=4, color="gray",
                label="random classifier")
    ax.bar_label(b1, fmt="%.1f%%", padding=3)
    ax.bar_label(b2, fmt="%.1f%%", padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("BoW model vs random classifier (mean +/- std over runs)")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = os.path.join("stl10", "stl10_raw")
    train_dir, test_dir = os.path.join(root, "train"), os.path.join(root, "test")
    n_points_x = n_points_y = 10
    N_RUNS, N_ITER = 10, 10
    K_LIST = [50, 100, 200, 400]
    METRICS = ("euclidean", "cityblock")  # L2 and L1

    print("Extracting / loading descriptors...")
    train_desc, train_labels = compute_descriptors_set(train_dir, n_points_x, n_points_y)
    test_desc, test_labels = compute_descriptors_set(test_dir, n_points_x, n_points_y)

    # --- the two settings -------------------------------------------------
    # 2 classes (bird = 1, horse = 6): all their images
    tr2_d, tr2_l = select_classes(train_desc, train_labels, [1, 6])
    te2_d, te2_l = select_classes(test_desc, test_labels, [1, 6])
    # 10 classes: 10% of the images of every class (fixed seed -> same subset everywhere)
    tr10_d, tr10_l = subsample_per_class(train_desc, train_labels, 0.1, seed=0)
    te10_d, te10_l = subsample_per_class(test_desc, test_labels, 0.1, seed=0)

    settings = {
        "2 classes (bird vs horse)": (tr2_d, tr2_l, te2_d, te2_l, 2),
        "10 classes (10% of images)": (tr10_d, tr10_l, te10_d, te10_l, 10),
    }
    for name, (trd, trl, ted, tel, C) in settings.items():
        print(f"{name}: train {trd.shape[0]} images, test {ted.shape[0]} images")

    # --- run everything ---------------------------------------------------
    results, baselines = {}, {}
    for name, (trd, trl, ted, tel, C) in settings.items():
        baselines[name] = run_random_baseline(tel, C, N_RUNS)
        for K in K_LIST:
            print(f"Running {name}, K={K} ...", flush=True)
            results[(name, K)] = run_experiment(trd, trl, ted, tel, K,
                                                metrics=METRICS, n_runs=N_RUNS, n_iter=N_ITER)

    # --- report -----------------------------------------------------------
    print("\n=== Experiments 1 & 2: K=100, L2 (mean +/- std over %d runs) ===" % N_RUNS)
    for name in settings:
        print(f"{name:28s} BoW model: {fmt(results[(name, 100)]['euclidean'])}"
              f"   random classifier: {fmt(baselines[name])}")

    print("\n=== Experiment 3: L1 vs L2 for different K ===")
    for name in settings:
        print(f"\n{name}   (random classifier: {fmt(baselines[name])})")
        print(f"{'K':>5s} | {'L2':>18s} | {'L1':>18s}")
        for K in K_LIST:
            r = results[(name, K)]
            print(f"{K:5d} | {fmt(r['euclidean']):>18s} | {fmt(r['cityblock']):>18s}")

    # --- save raw results (so the plots can be redone without re-running) and plot --------
    import pickle

    with open("results.pkl", "wb") as f:
        pickle.dump({"results": results, "baselines": baselines, "k_list": K_LIST}, f)
    plot_model_vs_random(results, baselines, K=100, show=False)
    plot_results(results, baselines, K_LIST, show=True)
# ----------------------------------------------------------------------
# Main: sanity check of the whole pipeline (experiments come next) and generation fo the HOG descriptor in the cache folder for the first run
# ----------------------------------------------------------------------
# if __name__ == "__main__":
    # root = os.path.join("STL10", "stl10_raw")
    # train_dir, test_dir = os.path.join(root, "train"), os.path.join(root, "test")
    # n_points_x = n_points_y = 10
    #
    # print("Extracting / loading descriptors...")
    # train_desc, train_labels = compute_descriptors_set(train_dir, n_points_x, n_points_y)
    # test_desc, test_labels = compute_descriptors_set(test_dir, n_points_x, n_points_y)
    # print(train_desc.shape, test_desc.shape)
    #
    # # one run, 10 classes, K=100, 10 iterations, L2
    # K = 100
    # codebook = kmeans(train_desc.reshape(-1, 128), K, max_iter=10, seed=0)
    # train_hist = bow_histograms(train_desc, codebook)
    # test_hist = bow_histograms(test_desc, codebook)
    # pred = nearest_neighbor_predict(test_hist, train_hist, train_labels, metric="euclidean")
    # print(f"Accuracy (10 classes, K={K}, L2): {accuracy(pred, test_labels) * 100:.2f}%")