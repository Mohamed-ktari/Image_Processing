import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter1d, gaussian_filter, convolve, rotate, zoom


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def load_gray(path):
    """Load an image and return it as a float grayscale array in [0, 255]."""
    img = mpimg.imread(path).astype(np.float64)
    if img.ndim == 3:                       # RGB(A) -> gray
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    if img.max() <= 1.0:                    # PNG are read in [0,1]
        img = img * 255.0
    return img


# ---------------------------------------------------------------
# Step 1 : image derivatives Ix, Iy
# ---------------------------------------------------------------
def compute_derivatives(I, method="gaussian", sigma_d=1.0):
    """
    method = 'gaussian' : convolve I with the partial derivatives of a 2D Gaussian
                          (separable: derivative along one axis, smoothing along the other)
    method = 'sobel'    : horizontal / vertical Sobel masks
    Returns Ix (derivative along columns) and Iy (derivative along rows).
    """
    if method == "gaussian":
        Ix = gaussian_filter1d(gaussian_filter1d(I, sigma_d, axis=0, order=0),
                               sigma_d, axis=1, order=1)
        Iy = gaussian_filter1d(gaussian_filter1d(I, sigma_d, axis=1, order=0),
                               sigma_d, axis=0, order=1)
    elif method == "sobel":
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float64)
        sobel_y = sobel_x.T
        Ix = convolve(I, sobel_x, mode="reflect")
        Iy = convolve(I, sobel_y, mode="reflect")
    else:
        raise ValueError("method must be 'gaussian' or 'sobel'")
    return Ix, Iy


# ---------------------------------------------------------------
# Step 2 : second-order moments
# ---------------------------------------------------------------
def compute_moments(Ix, Iy):
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    return Ix2, Iy2, Ixy


# ---------------------------------------------------------------
# Step 3 : smoothing with a second Gaussian (sigma_w > sigma_d)
# ---------------------------------------------------------------
def smooth_moments(Ix2, Iy2, Ixy, sigma_w=2.0):
    Ix2_s = gaussian_filter(Ix2, sigma_w)
    Iy2_s = gaussian_filter(Iy2, sigma_w)
    Ixy_s = gaussian_filter(Ixy, sigma_w)
    return Ix2_s, Iy2_s, Ixy_s


# ---------------------------------------------------------------
# Step 4 : Harris response H = det(A) - k * trace(A)^2
# ---------------------------------------------------------------
def harris_response(Ix2_s, Iy2_s, Ixy_s, k=0.04):
    det_A = Ix2_s * Iy2_s - Ixy_s ** 2
    trace_A = Ix2_s + Iy2_s
    return det_A - k * trace_A ** 2


# ---------------------------------------------------------------
# Step 5 : non-maximum suppression + threshold
# ---------------------------------------------------------------
def non_max_suppression(H, theta, window=3):
    """
    A pixel is a corner if H >= theta and H equals the max over its
    (window x window) neighbourhood (itself included).
    Returns an array of (row, col) corner coordinates.
    """
    r = window // 2
    padded = np.pad(H, r, mode="constant", constant_values=-np.inf)
    local_max = np.full_like(H, -np.inf)
    rows, cols = H.shape
    for di in range(2 * r + 1):
        for dj in range(2 * r + 1):
            shifted = padded[di:di + rows, dj:dj + cols]
            local_max = np.maximum(local_max, shifted)
    mask = (H >= theta) & (H == local_max)
    return np.argwhere(mask)


# ---------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------
def harris_detector(I, method="gaussian", sigma_d=1.0, sigma_w=2.0,
                    k=0.04, thresh_ratio=0.01, nms_window=3):
    Ix, Iy = compute_derivatives(I, method, sigma_d)              # Step 1
    Ix2, Iy2, Ixy = compute_moments(Ix, Iy)                       # Step 2
    Ix2_s, Iy2_s, Ixy_s = smooth_moments(Ix2, Iy2, Ixy, sigma_w)  # Step 3
    H = harris_response(Ix2_s, Iy2_s, Ixy_s, k)                   # Step 4
    theta = thresh_ratio * H.max()                                # theta > 0
    corners = non_max_suppression(H, theta, nms_window)           # Step 5
    return H, corners


# ---------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------
def show_results(I, H, corners, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(I, cmap="gray")
    axes[0].set_title("Original " + title)
    axes[1].imshow(H, cmap="gray")
    axes[1].set_title("Harris response H")
    axes[2].imshow(I, cmap="gray")
    axes[2].plot(corners[:, 1], corners[:, 0], "r+", markersize=8)
    axes[2].set_title(f"Detected corners ({len(corners)})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()


# ---------------------------------------------------------------
# Transformations for the robustness tests
# ---------------------------------------------------------------
def rotate_image(I, angle):
    return rotate(I, angle, reshape=True, mode="constant", cval=0.0)


def scale_image(I, factor):
    return zoom(I, factor, order=1)


def change_intensity(I, gain=1.0, offset=0.0):
    return np.clip(gain * I + offset, 0, 255)


def add_noise(I, sigma=10.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(I + rng.normal(0, sigma, I.shape), 0, 255)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == '__main__':
    directory= "images/"
    images = ["CircleLineRect.png", "zurlim.png"]

    # parameters (tune them per image)
    params = dict(method="gaussian", sigma_d=1.0, sigma_w=2.0,
                  k=0.04, thresh_ratio=0.01, nms_window=3)

    for path in images:
        I = load_gray(directory + path)

        # ---- 1. Basic detection: response + corners
        H, corners = harris_detector(I, **params)
        show_results(I, H, corners, title=path)

        # ---- 2. Robustness tests
        transforms = {
            "Rotation 30 deg":        rotate_image(I, 30),
            "Rotation 90 deg":        rotate_image(I, 90),
            "Scale x0.5":             scale_image(I, 0.5),
            "Scale x1.5":             scale_image(I, 1.5),
            "Brightness +50":         change_intensity(I, 1.0, 50),
            "Contrast x0.5":          change_intensity(I, 0.5, 0),
            "Gaussian noise (s=15)":  add_noise(I, 15),
        }
        for name, It in transforms.items():
            Ht, Ct = harris_detector(It, **params)
            show_results(It, Ht, Ct, title=f"{path} - {name}")

    plt.show()