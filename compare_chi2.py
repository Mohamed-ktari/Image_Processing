"""
Comparison of the distances used by the 1-NN classifier: L2, L1 and chi-square.
Same protocol as the main experiments (10 runs, 10 K-means iterations, same seeds,
same data subsets). In each run the SAME codebook is used for the three distances,
so the comparison is paired and the L2 / L1 columns reproduce the earlier results.

Run from the same folder as BagOfWords_Classification.py (it reuses the descriptor cache).
"""
import os
import numpy as np
from BagOfWords_Classification import (
    list_images, compute_descriptors_set, select_classes, subsample_per_class,
    run_experiment, fmt,
)

PAIR = ("bird", "horse")             # the two classes of the 2-class setting
K_LIST = [50, 100, 200, 400]
METRICS = ("euclidean", "cityblock", "chi2")
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
            rows.append((K, run_experiment(trd, trl, ted, tel, K, metrics=METRICS,
                                           n_runs=N_RUNS, n_iter=N_ITER)))

        print(f"\n{name}")
        print(f"{'K':>5s} | {'L2':>18s} | {'L1':>18s} | {'chi-square':>18s} | "
              f"{'chi2 - L2':>13s} | {'chi2 - L1':>13s}")
        for K, r in rows:
            g2 = (r["chi2"] - r["euclidean"]) * 100
            g1 = (r["chi2"] - r["cityblock"]) * 100
            print(f"{K:5d} | {fmt(r['euclidean']):>18s} | {fmt(r['cityblock']):>18s} | "
                  f"{fmt(r['chi2']):>18s} | {g2.mean():+6.2f}+/-{g2.std():4.2f} | "
                  f"{g1.mean():+6.2f}+/-{g1.std():4.2f}")