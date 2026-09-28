import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter, convolve


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def load_gray(path):
    """Load an image (png, pgm, ...) as a float grayscale array in [0, 255]."""
    raw = mpimg.imread(path)
    img = raw.astype(np.float64)
    if img.ndim == 3:                                   # RGB(A) -> gray
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    if raw.dtype != np.uint8 and img.max() <= 1.0:      # float images in [0,1]
        img = img * 255.0
    return img


# ---------------------------------------------------------------
# Step 1 : Gaussian smoothing
# ---------------------------------------------------------------
def smooth_image(I, sigma=1.5):
    return gaussian_filter(I, sigma)


# ---------------------------------------------------------------
# Step 2 : Sobel derivatives of the smoothed image
# ---------------------------------------------------------------
def sobel_derivatives(I_smooth):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    sobel_y = sobel_x.T
    Ix = convolve(I_smooth, sobel_x, mode="reflect")   # derivative along columns
    Iy = convolve(I_smooth, sobel_y, mode="reflect")   # derivative along rows
    return Ix, Iy


# ---------------------------------------------------------------
# Step 3 : gradient magnitude, direction, and simple thresholding
# ---------------------------------------------------------------
def gradient_magnitude_direction(Ix, Iy):
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    direction = np.arctan2(Iy, Ix)          # in (-pi, pi], arctan2 avoids division by 0
    return magnitude, direction


def choose_threshold(magnitude, percentile=90):
    """Threshold adapted to the distribution of the gradient magnitudes."""
    return np.percentile(magnitude, percentile)


def threshold_edges(magnitude, threshold):
    return magnitude >= threshold


# ---------------------------------------------------------------
# Step 4 : non-maximum suppression along the gradient direction
# ---------------------------------------------------------------
def non_max_suppression(magnitude, direction, threshold=0.0):
    """
    The gradient is orthogonal to the edge, so a pixel is kept only if its magnitude
    is >= that of its two neighbours along the gradient direction.
    The direction is quantised into 4 orientations (0, 45, 90, 135 degrees).
    (>= is used instead of > so that plateaus of equal values do not vanish.)
    """
    rows, cols = magnitude.shape
    padded = np.pad(magnitude, 1, mode="constant", constant_values=0)

    def neighbour(di, dj):
        # value of the pixel at offset (di, dj) for every pixel
        return padded[1 + di:1 + di + rows, 1 + dj:1 + dj + cols]

    angle = np.rad2deg(direction) % 180.0           # [0, 180)
    # quantisation (row axis points down, so 45 deg -> (+1,+1) and 135 deg -> (+1,-1))
    q0 = (angle < 22.5) | (angle >= 157.5)          # gradient horizontal
    q45 = (angle >= 22.5) & (angle < 67.5)
    q90 = (angle >= 67.5) & (angle < 112.5)         # gradient vertical
    q135 = (angle >= 112.5) & (angle < 157.5)

    n1 = np.zeros_like(magnitude)
    n2 = np.zeros_like(magnitude)
    for mask, (di, dj) in ((q0, (0, 1)), (q45, (1, 1)), (q90, (1, 0)), (q135, (1, -1))):
        n1[mask] = neighbour(di, dj)[mask]
        n2[mask] = neighbour(-di, -dj)[mask]

    keep = (magnitude >= n1) & (magnitude >= n2) & (magnitude >= threshold)
    return keep


# ---------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------
def canny_detector(I, sigma=1.5, percentile=90):
    I_smooth = smooth_image(I, sigma)                              # Step 1
    Ix, Iy = sobel_derivatives(I_smooth)                           # Step 2
    magnitude, direction = gradient_magnitude_direction(Ix, Iy)    # Step 3
    threshold = choose_threshold(magnitude, percentile)
    edges_thresh = threshold_edges(magnitude, threshold)
    edges_nms = non_max_suppression(magnitude, direction, threshold)  # Step 4
    return dict(smooth=I_smooth, magnitude=magnitude, direction=direction,
                threshold=threshold, edges_thresh=edges_thresh, edges_nms=edges_nms)


# ---------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------
def show_results(I, res, title=""):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    axes[0].imshow(I, cmap="gray")
    axes[0].set_title("Original " + title)
    axes[1].imshow(res["smooth"], cmap="gray")
    axes[1].set_title("Gaussian-smoothed image")
    axes[2].imshow(res["magnitude"], cmap="gray")
    axes[2].set_title("Gradient magnitude")
    im = axes[3].imshow(res["direction"], cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[3].set_title("Gradient direction (rad)")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    axes[4].imshow(res["edges_thresh"], cmap="gray")
    axes[4].set_title(f"Thresholded magnitude (T={res['threshold']:.1f})")
    axes[5].imshow(res["edges_nms"], cmap="gray")
    axes[5].set_title("After non-maximum suppression")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()


def show_histogram(magnitude, threshold, title=""):
    plt.figure(figsize=(6, 4))
    plt.hist(magnitude.ravel(), bins=100, log=True)
    plt.axvline(threshold, color="r", linestyle="--", label=f"threshold = {threshold:.1f}")
    plt.title("Gradient magnitude histogram " + title)
    plt.xlabel("magnitude")
    plt.ylabel("count (log)")
    plt.legend()
    plt.tight_layout()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == '__main__':
    directory = "images/"
    images = ["zurlim.png", "cube_left.pgm", "cube_right.pgm"]

    sigma = 1.5         # Gaussian std of step 1
    percentile = 90     # keep the strongest 10% of gradient magnitudes (adjust per image)

    for path in images:
        I = load_gray(directory + path)
        res = canny_detector(I, sigma=sigma, percentile=percentile)
        show_results(I, res, title=path)
        show_histogram(res["magnitude"], res["threshold"], title=path)

    plt.show()