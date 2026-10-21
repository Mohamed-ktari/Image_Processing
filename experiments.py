"""Parameter study for the particle filter tracker.

Needs particle_filter.py in the same folder.
Run:  python experiments.py  (set VIDEO below, then select the object once)
"""
import time
import csv
import itertools
import numpy as np
import cv2
import matplotlib.pyplot as plt

from particle_filter import (compute_histogram, histogram_distance, init_particles,
                             predict, correct, estimate, resample,
                             effective_sample_size)

DEFAULTS = dict(sigma=15.0, lam=20.0, nbins=8, N=300, method="systematic")


# ----------------------------------------------------------------------
# One tracking run (headless) with a given set of parameters
# ----------------------------------------------------------------------
def run_tracker(video, center, size, sigma=15.0, lam=20.0, nbins=8, N=300,
                method="systematic", max_frames=None, seed=0,
                save_video=None, lost_thresh=0.6):
    """Track the object in `video` and return (trajectory, metrics).

    metrics:
      mean_dist  : mean histogram distance between the estimated area and h_ref
                   (lower = the estimate looks more like the target)
      lost_ratio : fraction of frames where that distance > lost_thresh
      mean_neff  : mean effective sample size (before resampling)
      mean_step  : mean displacement between consecutive estimates (jitter)
      fps        : processing speed
    """
    np.random.seed(seed)
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    if not ok:
        raise IOError(f"Cannot read {video}")

    h_ref = compute_histogram(frame, center, size, nbins)
    particles, weights = init_particles(center, N, sigma)

    writer = None
    if save_video:
        H, W = frame.shape[:2]
        writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))

    traj, dists, neffs = [np.asarray(center, float)], [0.0], []
    t0, n = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        particles = predict(particles, sigma, frame.shape)
        weights = correct(frame, particles, weights, size, h_ref, nbins, lam)
        c = estimate(particles, weights)
        neffs.append(effective_sample_size(weights))
        particles, weights = resample(particles, weights, method)

        h_c = compute_histogram(frame, c, size, nbins)
        dists.append(histogram_distance(h_c, h_ref) if h_c is not None else 1.0)
        traj.append(c)
        n += 1

        if writer is not None:
            for p in particles:
                cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 255, 255), -1)
            w, h = size
            cv2.rectangle(frame, (int(c[0] - w / 2), int(c[1] - h / 2)),
                          (int(c[0] + w / 2), int(c[1] + h / 2)), (0, 255, 0), 2)
            writer.write(frame)

    elapsed = max(time.time() - t0, 1e-9)
    cap.release()
    if writer is not None:
        writer.release()

    traj = np.array(traj)
    dists = np.array(dists)
    metrics = dict(
        mean_dist=float(dists.mean()),
        lost_ratio=float((dists > lost_thresh).mean()),
        mean_neff=float(np.mean(neffs)) if neffs else float("nan"),
        mean_step=float(np.linalg.norm(np.diff(traj, axis=0), axis=1).mean()) if len(traj) > 1 else 0.0,
        fps=n / elapsed,
    )
    return traj, metrics


# ----------------------------------------------------------------------
# Parameter study
# ----------------------------------------------------------------------
def test_parameters(video, center, size, grids=None, defaults=None, mode="one_at_a_time",
                    max_frames=None, n_repeats=3, save_dir=None, ground_truth=None):
    """Test the tracker with different settings of sigma, lambda, Nb, N and of the
    resampling algorithm (systematic, multinomial, residual).

    grids : dict param -> list of values to test. Default values below.
    mode  : "one_at_a_time" -> vary one parameter, keep the others at their default
            "full_grid"     -> every combination (can be long!)
    n_repeats : each setting is run several times (different seeds) and averaged,
                because the filter is stochastic.
    ground_truth : optional array (n_frames+1, 2) of true centres -> adds an RMSE column.
    Returns a list of result dicts (also saved as results.csv if save_dir is given).
    """
    base = dict(DEFAULTS, **(defaults or {}))
    grids = grids or dict(
        sigma=[5, 15, 30, 60],
        lam=[1, 5, 20, 50, 100],
        nbins=[2, 4, 8, 16],
        N=[20, 50, 100, 300, 1000],
        method=["systematic", "multinomial", "residual"],
    )

    # ---- build the list of configurations
    if mode == "full_grid":
        keys = list(grids)
        configs = [dict(base, **dict(zip(keys, vals)))
                   for vals in itertools.product(*[grids[k] for k in keys])]
    else:
        configs = [dict(base)]
        for k, values in grids.items():
            for v in values:
                cfg = dict(base, **{k: v})
                if cfg not in configs:
                    configs.append(cfg)

    # ---- run them
    results, trajectories = [], []
    for i, cfg in enumerate(configs):
        runs = []
        for r in range(n_repeats):
            vid = None
            if save_dir and r == 0:
                vid = f"{save_dir}/run{i:03d}.mp4"
            traj, m = run_tracker(video, center, size, max_frames=max_frames,
                                  seed=r, save_video=vid, **cfg)
            if ground_truth is not None:
                L = min(len(traj), len(ground_truth))
                m["rmse"] = float(np.sqrt(np.mean(
                    np.sum((traj[:L] - ground_truth[:L]) ** 2, axis=1))))
            runs.append(m)
            if r == 0:
                trajectories.append(traj)
        avg = {k: float(np.mean([m[k] for m in runs])) for k in runs[0]}
        results.append(dict(cfg, **avg))
        print(f"[{i + 1}/{len(configs)}] {cfg} -> "
              f"dist={avg['mean_dist']:.3f} lost={avg['lost_ratio']:.2f} "
              f"neff={avg['mean_neff']:.1f} fps={avg['fps']:.1f}")

    _print_table(results)
    if save_dir:
        _save_csv(results, f"{save_dir}/results.csv")
    plot_results(results, trajectories, base, grids, mode, save_dir)
    return results


def _print_table(results):
    cols = list(results[0].keys())
    print("\n" + " | ".join(f"{c:>10}" for c in cols))
    for r in results:
        print(" | ".join(f"{r[c]:>10.3f}" if isinstance(r[c], float) else f"{str(r[c]):>10}"
                         for c in cols))


def _save_csv(results, path):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader()
        wr.writerows(results)


def plot_results(results, trajectories, base, grids, mode="one_at_a_time", save_dir=None):
    """One figure per parameter (one_at_a_time mode): metric curves + trajectories."""
    if mode != "one_at_a_time":
        return
    for k, values in grids.items():
        sel = [(i, r) for i, r in enumerate(results)
               if all(r[p] == base[p] for p in grids if p != k)]
        sel.sort(key=lambda t: values.index(t[1][k]))
        labels = [str(r[k]) for _, r in sel]

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].bar(labels, [r["mean_dist"] for _, r in sel], color="tab:blue")
        ax[0].set_title(f"Mean histogram distance vs {k}")
        ax[0].set_xlabel(k)
        ax[1].bar(labels, [r["mean_neff"] for _, r in sel], color="tab:orange")
        ax[1].set_title(f"Mean effective sample size vs {k}")
        ax[1].set_xlabel(k)
        for (i, r), lab in zip(sel, labels):
            t = trajectories[i]
            ax[2].plot(t[:, 0], t[:, 1], label=f"{k}={lab}")
        ax[2].invert_yaxis()                       # image coordinates
        ax[2].set_title("Estimated trajectories")
        ax[2].legend(fontsize=7)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/study_{k}.png", dpi=120)
    plt.show()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    VIDEO = "video.mp4"               # <- your video
    OUT = "results" 
    os.makedirs(OUT, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Cannot read the video.")
    x, y, w, h = cv2.selectROI("Select object", first, False)
    cv2.destroyAllWindows()
    center, size = (x + w / 2, y + h / 2), (int(w), int(h))

    test_parameters(VIDEO, center, size, max_frames=None, n_repeats=3, save_dir=OUT)