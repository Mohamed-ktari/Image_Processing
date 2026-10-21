"""
Comparison: plain BoW vs BoW with visual TF-IDF weighting.
Same protocol as the main experiments (10 runs, 10 K-means iterations, same seeds,
same data subsets), same codebook for both variants in each run.

Run from the same folder as BagOfWords_Classification.py (it reuses the descriptor cache).
"""
import os
import numpy as np
from BagOfWords_Classification import (
    list_images, compute_descriptors_set, select_classes, subsample_per_class,
    run_experiment_tfidf, fmt,
)

PAIR = ("bird", "horse")          # the two classes of the 2-class setting
K_LIST = [50, 100, 200, 400]
METRICS = ("euclidean", "cityblock")  # L2, L1
N_RUNS, N_ITER = 10, 10

if __name__ == "__main__":
    root = os.path.join("STL10", "stl10_raw")
    train_dir, test_dir = os.path.join(root, "train"), os.path.join(root, "test")

    print("Loading descriptors...")
    train_desc, train_labels = compute_descriptors_set(train_dir, 10, 10)
    test_desc, test_labels = compute_descriptors_set(test_dir, 10, 10)

    _, _, class_names = list_images(train_dir)
    pair_ids = [class_names.index(c) for c in PAIR]
    tr2_d, tr2_l = select_classes(train_desc, train_labels, pair_ids)
    te2_d, te2_l = select_classes(test_desc, test_labels, pair_ids)
    tr10_d, tr10_l = subsample_per_class(train_desc, train_labels, 0.1, seed=0)
    te10_d, te10_l = subsample_per_class(test_desc, test_labels, 0.1, seed=0)

    settings = {
        f"2 classes ({PAIR[0]} vs {PAIR[1]})": (tr2_d, tr2_l, te2_d, te2_l),
        "10 classes (10% of images)": (tr10_d, tr10_l, te10_d, te10_l),
    }

    for name, (trd, trl, ted, tel) in settings.items():
        print(f"\n{name}: train {len(trl)} images, test {len(tel)} images")
        rows = []
        for K in K_LIST:
            print(f"  K={K} ...", flush=True)
            r = run_experiment_tfidf(trd, trl, ted, tel, K, metrics=METRICS,
                                     n_runs=N_RUNS, n_iter=N_ITER)
            rows.append((K, r))

        for metric, label in (("euclidean", "L2"), ("cityblock", "L1")):
            print(f"\n{name} - {label}")
            print(f"{'K':>5s} | {'baseline':>18s} | {'tf-idf':>18s} | {'gain (paired)':>16s}")
            for K, r in rows:
                b, t = r[("baseline", metric)], r[("tfidf", metric)]
                gain = (t - b) * 100
                print(f"{K:5d} | {fmt(b):>18s} | {fmt(t):>18s} | "
                      f"{gain.mean():+6.2f} +/- {gain.std():4.2f}")