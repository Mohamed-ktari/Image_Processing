import numpy as np
import cv2


# ----------------------------------------------------------------------
# Histogram and likelihood
# ----------------------------------------------------------------------
def extract_patch(frame, center, size):
    """Return the image patch of shape `size`=(w, h) centred at `center`=(x, y).
    The rectangle is clipped to the image borders."""
    H, W = frame.shape[:2]
    w, h = size
    x, y = int(round(center[0])), int(round(center[1]))
    x0, x1 = max(0, x - w // 2), min(W, x + w // 2)
    y0, y1 = max(0, y - h // 2), min(H, y + h // 2)
    if x1 <= x0 or y1 <= y0:          # rectangle completely outside the image
        return None
    return frame[y0:y1, x0:x1]


def compute_histogram(frame, center, size, nbins=8):
    """Normalised colour histogram (nbins per channel -> nbins^3 bins in total)
    of the rectangle centred at `center`. Returns None if the patch is empty."""
    patch = extract_patch(frame, center, size)
    if patch is None:
        return None
    hist = cv2.calcHist([patch], [0, 1, 2], None,
                        [nbins, nbins, nbins], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    s = hist.sum()
    return hist / s if s > 0 else None


def histogram_distance(h, h_ref):
    """Eq. (2): dist(h, h') = sqrt(1 - sum_i sqrt(h(i) h'(i)))."""
    bc = np.sum(np.sqrt(h * h_ref))               # Bhattacharyya coefficient
    return np.sqrt(max(0.0, 1.0 - bc))


def likelihood(frame, particle, size, h_ref, nbins=8, lam=20.0):
    """Eq. (3): g(y|x) ∝ exp(-lambda * dist^2(h_ref, h(x)))."""
    h = compute_histogram(frame, particle, size, nbins)
    if h is None:                                  # particle outside the image
        return 0.0
    d = histogram_distance(h, h_ref)
    return np.exp(-lam * d ** 2)


# ----------------------------------------------------------------------
# Particle filter steps
# ----------------------------------------------------------------------
def init_particles(center, N, sigma_init):
    """Step 3: N particles drawn around the initial centre, weights = 1/N."""
    center = np.asarray(center, dtype=float)
    particles = center + np.random.randn(N, 2) * sigma_init
    weights = np.full(N, 1.0 / N)
    return particles, weights


def predict(particles, sigma, img_shape=None):
    """Prediction, Eq. (1): X_k = X_{k-1} + V_k, V_k ~ N(0, Sigma) with Sigma diagonal.
    `sigma` is a scalar or a length-2 array of standard deviations."""
    noise = np.random.randn(*particles.shape) * np.asarray(sigma, dtype=float)
    particles = particles + noise
    if img_shape is not None:                      # keep particles inside the image
        H, W = img_shape[:2]
        particles[:, 0] = np.clip(particles[:, 0], 0, W - 1)
        particles[:, 1] = np.clip(particles[:, 1], 0, H - 1)
    return particles


def correct(frame, particles, weights, size, h_ref, nbins=8, lam=20.0):
    """Correction: w_k ∝ g(y_k | x_k) * w_{k-1}, then normalise."""
    g = np.array([likelihood(frame, p, size, h_ref, nbins, lam) for p in particles])
    new_w = g * weights
    s = new_w.sum()
    if s < 1e-300:                                 # all weights vanished -> reset
        return np.full(len(weights), 1.0 / len(weights))
    return new_w / s


def estimate(particles, weights):
    """Estimate of the position: x_bar = sum_i w_i x_i."""
    return np.sum(particles * weights[:, None], axis=0)


# ----------------------------------------------------------------------
# Resampling (each function returns the indices of the selected particles)
# ----------------------------------------------------------------------
def systematic_resampling(weights):
    N = len(weights)
    u = np.random.uniform(0, 1.0 / N) + np.arange(N) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0                               # avoid rounding problems
    return np.searchsorted(cumsum, u)


def multinomial_resampling(weights):
    N = len(weights)
    return np.random.choice(N, size=N, p=weights)


def residual_resampling(weights):
    N = len(weights)
    n_det = np.floor(N * weights).astype(int)      # deterministic copies
    indices = np.repeat(np.arange(N), n_det)
    n_rest = N - n_det.sum()
    if n_rest > 0:
        w_res = weights - n_det / N                # residual weights
        w_res = w_res / w_res.sum()
        extra = np.random.choice(N, size=n_rest, p=w_res)
        indices = np.concatenate([indices, extra])
    return indices


def resample(particles, weights, method="systematic"):
    """Resample particles and reset weights to 1/N."""
    methods = {"systematic": systematic_resampling,
               "multinomial": multinomial_resampling,
               "residual": residual_resampling}
    idx = methods[method](weights)
    return particles[idx], np.full(len(weights), 1.0 / len(weights))


def effective_sample_size(weights):
    """Optional: resample only when N_eff falls below a threshold (e.g. N/2)."""
    return 1.0 / np.sum(weights ** 2)


# ----------------------------------------------------------------------
# One full iteration of the tracker
# ----------------------------------------------------------------------
def track_step(frame, particles, weights, size, h_ref,
               sigma=15.0, nbins=8, lam=20.0, method="systematic"):
    """Prediction -> Correction -> estimate -> Resampling.
    Returns (particles, weights, estimated_center)."""
    particles = predict(particles, sigma, frame.shape)
    weights = correct(frame, particles, weights, size, h_ref, nbins, lam)
    center = estimate(particles, weights)          # estimate after correction
    particles, weights = resample(particles, weights, method)
    return particles, weights, center


# ----------------------------------------------------------------------
# Minimal driver (to be completed later)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    VIDEO = 0            # replace with a path, e.g. "video.mp4"
    N, NBINS, LAM, SIGMA = 300, 8, 20.0, 15.0

    cap = cv2.VideoCapture(VIDEO)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Cannot read the video.")

    # Step 1: user selects the rectangle
    x, y, w, h = cv2.selectROI("Select object", frame, False)
    cv2.destroyWindow("Select object")
    center, size = (x + w / 2, y + h / 2), (w, h)

    # Step 2: reference histogram
    h_ref = compute_histogram(frame, center, size, NBINS)

    # Step 3: particles
    particles, weights = init_particles(center, N, sigma_init=SIGMA)

    # Step 4: tracking loop
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        particles, weights, c = track_step(frame, particles, weights, size,
                                           h_ref, SIGMA, NBINS, LAM)
        for p in particles:
            cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 255, 255), -1)
        cv2.rectangle(frame,
                      (int(c[0] - w / 2), int(c[1] - h / 2)),
                      (int(c[0] + w / 2), int(c[1] + h / 2)), (0, 255, 0), 2)
        cv2.imshow("Particle filter tracker", frame)
        if cv2.waitKey(30) & 0xFF == 27:           # Esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()