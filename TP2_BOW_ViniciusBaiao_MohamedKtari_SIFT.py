import os
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# Reading the class names from a text file
def read_class_names(path):
    with open(path) as f:
        class_names = [line.strip() for line in f if line.strip() != ""]
    return class_names


# Loading the images in greyscale of one class
def load_images_from_folder(folder, fraction=1.0):
    filenames = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))

    if fraction < 1.0:
        n_keep = int(len(filenames) * fraction)
        chosen = np.random.choice(len(filenames), n_keep, replace=False)
        filenames = [filenames[i] for i in sorted(chosen)]

    images = []
    for filename in filenames:
        img = Image.open(os.path.join(folder, filename))
        images.append(np.asarray(img.convert("L")))
    return images


# step 1: dense regular grid
def dense_regular_grid(image, n_points_x, n_points_y):
    height, width = image.shape[:2]

    x = np.rint(np.linspace(0, width - 1, n_points_x)).astype(int)
    y = np.rint(np.linspace(0, height - 1, n_points_y)).astype(int)

    x_grid, y_grid = np.meshgrid(x, y)
    return x_grid.ravel(), y_grid.ravel()

# SIFT-style normalisation: unit length, clip at 0.2, unit length again
def normalize_descriptors(descriptors):
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / (norms + 1e-7)
    descriptors = np.minimum(descriptors, 0.2)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / (norms + 1e-7)
    return descriptors

# step 2: HOG descriptors
def hog_features(image, x_indices, y_indices, use_SIFT_normalization=True):
    image = image.astype(float) # converting to float to avoid overflow in gradient computation

    # reflect padding so every point gets a full 16x16 patch centered on it, like for example the point (0, 0) which would need pixels from (-8, -8) to (7, 7)
    padded = np.pad(image, 8, mode="reflect")

    gx = ndimage.sobel(padded, axis=1)
    gy = ndimage.sobel(padded, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)  # angle in [-pi, pi], like the one done in the TP1

    descriptors = np.zeros((len(x_indices), 128))

    for point in range(len(x_indices)):
        x = x_indices[point]
        y = y_indices[point]

        # in the padded image, the patch centered on (x, y) starts at (y, x)
        patch_mag = magnitude[y:y + 16, x:x + 16]
        patch_ori = orientation[y:y + 16, x:x + 16]

        descriptor = []
        for cell_y in range(4):
            for cell_x in range(4):
                cell_mag = patch_mag[cell_y * 4:(cell_y + 1) * 4, cell_x * 4:(cell_x + 1) * 4]
                cell_ori = patch_ori[cell_y * 4:(cell_y + 1) * 4, cell_x * 4:(cell_x + 1) * 4]

                hist, _ = np.histogram(cell_ori, bins=8, range=(-np.pi, np.pi), weights=cell_mag)
                descriptor.extend(hist)

        descriptors[point] = descriptor

    if use_SIFT_normalization:
        descriptors = normalize_descriptors(descriptors)

    return descriptors


# k-means
def kmeans(data, k, n_iter):
    # step 1: random initial centroids, chosen among the data points
    indices = np.random.permutation(len(data))[:k]
    centroids = data[indices].copy()

    for iteration in range(n_iter):
        # assigning each point to its nearest centroid
        distances = cdist(data, centroids, metric="euclidean")
        assignments = np.argmin(distances, axis=1)

        # step 3: move each centroid to the mean of its points
        for j in range(k):
            points_in_cluster = data[assignments == j]
            if len(points_in_cluster) > 0:
                centroids[j] = points_in_cluster.mean(axis=0)

    return centroids


# BoW histograms
def bow_histograms(descriptors_per_image, labels, centroids):
    k = len(centroids)
    histograms = np.zeros((len(descriptors_per_image), k))

    for i in range(len(descriptors_per_image)):
        distances = cdist(descriptors_per_image[i], centroids, metric="euclidean")
        nearest_word = np.argmin(distances, axis=1)
        histograms[i] = np.bincount(nearest_word, minlength=k)

    return histograms, np.array(labels)


# nearest neighbor classifier
# metric="euclidean" is the L2 norm and if metric="cityblock" it means the L1 norm
def nearest_neighbor_classifier(train_histograms, train_labels, test_histograms, metric="euclidean"):
    distances = cdist(test_histograms, train_histograms, metric=metric)
    nearest = np.argmin(distances, axis=1)
    return train_labels[nearest]

# creating the random classifier to use as the baseline depending on the number of classes being used
def random_classifier(n_test, n_classes):
    return np.random.randint(0, n_classes, size=n_test)


if __name__ == "__main__":
    data_root = "STL10/stl10_raw"
    classes = ["bird", "horse"] # for the first case with 2 classes
    n_points_x = 8
    n_points_y = 8
    use_SIFT_normalization = True

    # Fixed seed so the random 20% subset (and the K-means runs) are the same every time
    np.random.seed(0)

    # Load the images and their labels (0 = airplane, 1 = car)
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for label, class_name in enumerate(classes):
        imgs = load_images_from_folder(os.path.join(data_root, "train", class_name))
        train_images += imgs
        train_labels += [label] * len(imgs)

        imgs = load_images_from_folder(os.path.join(data_root, "test", class_name))
        test_images += imgs
        test_labels += [label] * len(imgs)

    print("Train images:", len(train_images), " Test images:", len(test_images))

    # Grid points
    x_grid, y_grid = dense_regular_grid(train_images[0], n_points_x, n_points_y)

    train_descriptors = []
    test_descriptors = []

    # HOG descriptors for every image
    for img in train_images:
        train_descriptors.append(hog_features(img, x_grid, y_grid, use_SIFT_normalization))

    for img in test_images:
        test_descriptors.append(hog_features(img, x_grid, y_grid, use_SIFT_normalization))

    # All training descriptors stacked together, used for K-means
    all_train_descriptors = np.vstack(train_descriptors)
    print("Descriptors for K-means:", all_train_descriptors.shape)

    # Experiment 1: K = 100, L2 distance, 10 runs, compared with a random classifier
    K = 100
    n_iter = 10
    n_runs = 10

    print("\nExperiment 1:")
    bow_accuracies = []
    random_accuracies = []
    for run in range(n_runs):
        centroids = kmeans(all_train_descriptors, K, n_iter)
        train_hist, y_train = bow_histograms(train_descriptors, train_labels, centroids)
        test_hist, y_test = bow_histograms(test_descriptors, test_labels, centroids)

        predictions = nearest_neighbor_classifier(train_hist, y_train, test_hist, metric="euclidean")
        bow_accuracies.append(np.mean(predictions == y_test))

        random_predictions = random_classifier(len(y_test), len(classes))
        random_accuracies.append(np.mean(random_predictions == y_test))

        print(f"Run {run + 1}: BoW accuracy = {bow_accuracies[-1]:.4f}, random accuracy = {random_accuracies[-1]:.4f}")

    print(f"BoW (L2): mean = {np.mean(bow_accuracies):.4f}, std = {np.std(bow_accuracies):.4f}")
    print(f"Random:   mean = {np.mean(random_accuracies):.4f}, std = {np.std(random_accuracies):.4f}")

    # Experiment 2 using 20% of each class
    fraction_10_classes = 0.1 # use 20% of each class

    print("\nExperiment 2 using 20% of each class:")

    all_classes = read_class_names("STL10/stl10_binary/class_names.txt")
    print("Classes:", all_classes)

    train_images_10, train_labels_10 = [], []
    test_images_10, test_labels_10 = [], []

    for label, class_name in enumerate(all_classes):
        imgs = load_images_from_folder(os.path.join(data_root, "train", class_name), fraction_10_classes)
        train_images_10 += imgs
        train_labels_10 += [label] * len(imgs)

        imgs = load_images_from_folder(os.path.join(data_root, "test", class_name), fraction_10_classes)
        test_images_10 += imgs
        test_labels_10 += [label] * len(imgs)

    print("Train images:", len(train_images_10), " Test images:", len(test_images_10))

    train_descriptors_10 = []
    for img in train_images_10:
        train_descriptors_10.append(hog_features(img, x_grid, y_grid, use_SIFT_normalization))

    test_descriptors_10 = []
    for img in test_images_10:
        test_descriptors_10.append(hog_features(img, x_grid, y_grid, use_SIFT_normalization))

    all_train_descriptors_10 = np.vstack(train_descriptors_10)

    bow_accuracies_10 = []
    random_accuracies_10 = []

    for run in range(n_runs):
        centroids = kmeans(all_train_descriptors_10, K, n_iter)
        train_hist, y_train = bow_histograms(train_descriptors_10, train_labels_10, centroids)
        test_hist, y_test = bow_histograms(test_descriptors_10, test_labels_10, centroids)

        predictions = nearest_neighbor_classifier(train_hist, y_train, test_hist, metric="euclidean")
        bow_accuracies_10.append(np.mean(predictions == y_test))

        random_predictions = random_classifier(len(y_test), len(all_classes))
        random_accuracies_10.append(np.mean(random_predictions == y_test))

        print(f"Run {run + 1}: BoW accuracy = {bow_accuracies_10[-1]:.4f}, random accuracy = {random_accuracies_10[-1]:.4f}")

    print(f"BoW (L2): mean = {np.mean(bow_accuracies_10):.4f}, std = {np.std(bow_accuracies_10):.4f}")
    print(f"Random:   mean = {np.mean(random_accuracies_10):.4f}, std = {np.std(random_accuracies_10):.4f}")

    # Experiment 3 with only the 2 classes, doing the L1 vs L2 for different values of K
    K_values = [50, 100, 200, 400]

    print("\nExperiment 3 doing the L1 vs L2 for different K:")
    l2_means, l2_stds = [], []
    l1_means, l1_stds = [], []

    for k in K_values:
        l2_accuracies = []
        l1_accuracies = []
        for run in range(n_runs):
            centroids = kmeans(all_train_descriptors, k, n_iter)
            train_hist, y_train = bow_histograms(train_descriptors, train_labels, centroids)
            test_hist, y_test = bow_histograms(test_descriptors, test_labels, centroids)

            predictions_l2 = nearest_neighbor_classifier(train_hist, y_train, test_hist, metric="euclidean")
            predictions_l1 = nearest_neighbor_classifier(train_hist, y_train, test_hist, metric="cityblock")
            l2_accuracies.append(np.mean(predictions_l2 == y_test))
            l1_accuracies.append(np.mean(predictions_l1 == y_test))

        print(f"K = {k}: L2 mean = {np.mean(l2_accuracies):.4f} (std {np.std(l2_accuracies):.4f}), "
              f"L1 mean = {np.mean(l1_accuracies):.4f} (std {np.std(l1_accuracies):.4f})")

        l2_means.append(np.mean(l2_accuracies))
        l2_stds.append(np.std(l2_accuracies))
        l1_means.append(np.mean(l1_accuracies))
        l1_stds.append(np.std(l1_accuracies))

    # Plot for the 2 classes: accuracy vs K for L1 and L2
    plt.figure()
    plt.errorbar(K_values, l2_means, yerr=l2_stds, marker="o", capsize=4, label="L2")
    plt.errorbar(K_values, l1_means, yerr=l1_stds, marker="o", capsize=4, label="L1")
    plt.axhline(1 / len(classes), color="gray", linestyle="--", label="random (chance)")
    plt.xlabel("K (number of visual words)")
    plt.ylabel("Test accuracy")
    plt.title("BoW + nearest neighbor: L1 vs L2 (2 classes)")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_K_2_classes.png")

    # Experiment 3 with the 10 classes (20% of each class), L1 vs L2 for different values of K
    print("\nExperiment 3 with the 10 classes, doing the L1 vs L2 for different K:")
    l2_means_10, l2_stds_10 = [], []
    l1_means_10, l1_stds_10 = [], []

    for k in K_values:
        l2_accuracies = []
        l1_accuracies = []
        for run in range(n_runs):
            centroids = kmeans(all_train_descriptors_10, k, n_iter)
            train_hist, y_train = bow_histograms(train_descriptors_10, train_labels_10, centroids)
            test_hist, y_test = bow_histograms(test_descriptors_10, test_labels_10, centroids)

            predictions_l2 = nearest_neighbor_classifier(train_hist, y_train, test_hist, metric="euclidean")
            predictions_l1 = nearest_neighbor_classifier(train_hist, y_train, test_hist, metric="cityblock")
            l2_accuracies.append(np.mean(predictions_l2 == y_test))
            l1_accuracies.append(np.mean(predictions_l1 == y_test))

        print(f"K = {k}: L2 mean = {np.mean(l2_accuracies):.4f} (std {np.std(l2_accuracies):.4f}), "
              f"L1 mean = {np.mean(l1_accuracies):.4f} (std {np.std(l1_accuracies):.4f})")

        l2_means_10.append(np.mean(l2_accuracies))
        l2_stds_10.append(np.std(l2_accuracies))
        l1_means_10.append(np.mean(l1_accuracies))
        l1_stds_10.append(np.std(l1_accuracies))

    # Plot for the 10 classes: accuracy vs K for L1 and L2
    plt.figure()
    plt.errorbar(K_values, l2_means_10, yerr=l2_stds_10, marker="o", capsize=4, label="L2")
    plt.errorbar(K_values, l1_means_10, yerr=l1_stds_10, marker="o", capsize=4, label="L1")
    plt.axhline(1 / len(all_classes), color="gray", linestyle="--", label="random (chance)")
    plt.xlabel("K (number of visual words)")
    plt.ylabel("Test accuracy")
    plt.title("BoW + nearest neighbor: L1 vs L2 (10 classes, 20% of each)")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_K_10_classes.png")

    # Show both figures at the end
    plt.show()
