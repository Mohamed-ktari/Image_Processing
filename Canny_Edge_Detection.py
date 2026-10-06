import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter, convolve


# utilities
def load_gray(path):
    raw = mpimg.imread(path)
    img = raw.astype(np.float64)
    if img.ndim == 3:                               
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    if raw.dtype != np.uint8 and img.max() <= 1.0:
        img = img * 255.0
    return img


# step 1: Gaussian smoothing
def smooth_image(I, sigma=1.5):
    return gaussian_filter(I, sigma)


# step 2: Sobel derivatives of the smoothed image
def sobel_derivatives(I_smooth):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    sobel_y = sobel_x.T
    Ix = convolve(I_smooth, sobel_x, mode="reflect")   # derivative along columns
    Iy = convolve(I_smooth, sobel_y, mode="reflect")   # derivative along rows
    return Ix, Iy


# step 3: gradient magnitude, direction, and simple thresholding
def gradient_magnitude_direction(Ix, Iy):
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    direction = np.arctan2(Iy, Ix)
    return magnitude, direction


def choose_threshold(magnitude, percentile=90):
    # threshold adapted to the distribution of the gradient magnitudes
    return np.percentile(magnitude, percentile)


def threshold_edges(magnitude, threshold):
    return magnitude >= threshold


# step 4 : non-maximum suppression along the gradient direction
def non_max_suppression(magnitude, direction, threshold=0.0):
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


# full pipeline
def canny_detector(I, sigma=1.5, percentile=90):
    I_smooth = smooth_image(I, sigma)                              # step 1
    Ix, Iy = sobel_derivatives(I_smooth)                           # step 2
    magnitude, direction = gradient_magnitude_direction(Ix, Iy)    # step 3
    threshold = choose_threshold(magnitude, percentile)
    edges_thresh = threshold_edges(magnitude, threshold)
    edges_nms = non_max_suppression(magnitude, direction, threshold)  # step 4
    return dict(smooth=I_smooth, magnitude=magnitude, direction=direction,
                threshold=threshold, edges_thresh=edges_thresh, edges_nms=edges_nms)


# visualisation
def show_results_interactive(I, res, title=""):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plt.subplots_adjust(bottom=0.15)
    axes = axes.ravel()
    
    # plotting static images
    axes[0].imshow(I, cmap="gray")
    axes[0].set_title("Original " + title)
    
    axes[1].imshow(res["smooth"], cmap="gray")
    axes[1].set_title("Gaussian-smoothed image")
    
    axes[2].imshow(res["magnitude"], cmap="gray")
    axes[2].set_title("Gradient magnitude")
    
    im = axes[3].imshow(res["direction"], cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[3].set_title("Gradient direction (rad)")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    
    # plotting dynamic images
    img_edges_thresh = axes[4].imshow(res["edges_thresh"], cmap="gray")
    axes[4].set_title(f"Thresholded magnitude (T={res['threshold']:.1f})")
    
    img_edges_nms = axes[5].imshow(res["edges_nms"], cmap="gray")
    axes[5].set_title("After non-maximum suppression")
    
    for ax in axes:
        ax.axis("off")
        
    # Slider to see the effect of the threshold in the images
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])

    slider = Slider(
        ax=ax_slider,
        label='Percentile',
        valmin=50.0,
        valmax=99.9,
        valinit=90.0
    )
    
    # the update function called when the slider is moved
    def update(val):
        current_percentile = slider.val
        
        # recalculate the threshold and dependent images using existing magnitude/direction
        new_threshold = choose_threshold(res["magnitude"], current_percentile)
        new_edges_thresh = threshold_edges(res["magnitude"], new_threshold)
        new_edges_nms = non_max_suppression(res["magnitude"], res["direction"], new_threshold)
        
        # update the image data
        img_edges_thresh.set_data(new_edges_thresh)
        img_edges_nms.set_data(new_edges_nms)
        
        # update the title with the new absolute threshold value
        axes[4].set_title(f"Thresholded magnitude (T={new_threshold:.1f})")
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    return slider


def show_histogram(magnitude, threshold, title=""):
    plt.figure(figsize=(6, 4))
    plt.hist(magnitude.ravel(), bins=100, log=True)
    plt.axvline(threshold, color="r", linestyle="--", label=f"threshold = {threshold:.1f}")
    plt.title("Gradient magnitude histogram " + title)
    plt.xlabel("magnitude")
    plt.ylabel("count (log)")
    plt.legend()
    plt.tight_layout()


if __name__ == '__main__':
    directory = "images/"
    images = ["zurlim.png", "cube_left.pgm", "cube_right.pgm"]

    sigma = 1.5         
    percentile = 90.0    

    active_sliders = []

    for path in images:
        I = load_gray(directory + path)
        res = canny_detector(I, sigma=sigma, percentile=percentile)
        
        slider = show_results_interactive(I, res, title=path)
        active_sliders.append(slider)

        show_histogram(res["magnitude"], res["threshold"], title=path)
            

    plt.show()