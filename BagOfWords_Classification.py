import numpy as np
from scipy import ndimage


def grid_points(img, n_points_x, n_points_y, border=8):
    # take the height and the width of the image
    h, w = img.shape[:2]
    xs = np.linspace(border, w - border, n_points_x).astype(int)
    ys = np.linspace(border, h - border, n_points_y).astype(int)

    x_grid, y_grid = np.meshgrid(xs, ys)   # both of shape (n_points_y, n_points_x)
    return x_grid.ravel(), y_grid.ravel()


def descriptors_hog(img, x_coords, y_coords, cell_size=4, n_cells=4, n_bins=8):
    """
    Step 2: HOG descriptors at the given grid points.

    Each point -> 16x16 patch (n_cells x n_cells cells of cell_size x cell_size pixels),
    8-bin histogram of gradient directions per cell, concatenated into a 128-D vector.

    Parameters
    ----------
    img : 2D array (H x W), grayscale image
    x_coords, y_coords : 1D arrays from grid_points()

    Returns
    -------
    descriptors : array of shape (n_points, n_cells * n_cells * n_bins) = (n_points, 128)
    """
    img = img.astype(np.float64)
    if img.ndim == 3:                       # safety: convert RGB to gray
        img = img.mean(axis=2)

    # Sobel gradients (axis=1 -> x direction, axis=0 -> y direction)
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    angles = np.arctan2(gy, gx)             # direction in [-pi, pi]

    patch_size = cell_size * n_cells        # 16
    half = patch_size // 2                  # 8
    h, w = img.shape

    n_points = len(x_coords)
    descriptors = np.zeros((n_points, n_cells * n_cells * n_bins))

    for p, (x, y) in enumerate(zip(x_coords, y_coords)):
        # patch centered at (x, y); clipped to stay inside the image
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
        descriptors[p] = np.concatenate(feat)

    return descriptors

import numpy as np
from scipy.spatial.distance import cdist


def kmeans(data, K, max_iter=100, tol=1e-4, seed=None):
    """
    K-means clustering to build the visual codebook.

    Parameters
    ----------
    data : array of shape (N, 128), all HOG descriptors of the training set
    K : number of clusters (visual words)
    max_iter : maximum number of iterations
    tol : convergence threshold on the centroid displacement
    seed : optional random seed for reproducibility

    Returns
    -------
    centroids : array of shape (K, 128), the final cluster centroids
    """
    if seed is not None:
        np.random.seed(seed)

    N = data.shape[0]

    # Step 1: initialize centroids with K distinct random data points
    perm = np.random.permutation(N)
    centroids = data[perm[:K]].astype(np.float64).copy()

    for it in range(max_iter):
        # Step 2: assign each point to its nearest centroid (Euclidean distance)
        dists = cdist(data, centroids, metric="euclidean")   # (N, K)
        labels = np.argmin(dists, axis=1)                     # (N,)

        # Step 3: update each centroid to the mean of its assigned points
        new_centroids = centroids.copy()
        for k in range(K):
            members = data[labels == k]
            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)
            else:
                # empty cluster: re-initialize with a random data point
                new_centroids[k] = data[np.random.randint(N)]

        # Convergence check: centroids barely moved
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return centroids




if __name__ == "main":
    pass