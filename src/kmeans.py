import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from skimage.io import imread, imsave
from tkinter import Tk, filedialog, simpledialog, messagebox

# Function to open a file dialog and select a dataset directory
def select_dataset_directory():
    root = Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="D:\\internship\\pro2\\train")
    return folder_path

# Function to open a file dialog and select an image file from a given directory
def select_image_file(folder_path):
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        initialdir=folder_path, 
        title="Select an Image File", 
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    return file_path

# Function to prompt the user for the number of clusters
def get_number_of_clusters():
    root = Tk()
    root.withdraw()
    try:
        n_clusters = simpledialog.askinteger("Input", "Enter the number of clusters (e.g., 2-10):", minvalue=2, maxvalue=10)
        return n_clusters
    except Exception:
        messagebox.showerror("Error", "Invalid number of clusters.")
        return None

# Function to normalize an image
def normalize_image(image):
    scaler = MinMaxScaler()
    reshaped = image.reshape(-1, 3)  # Reshape for normalization
    normalized = scaler.fit_transform(reshaped).reshape(image.shape)
    return normalized, scaler

# Function to denormalize an image
def denormalize_image(image, scaler):
    reshaped = image.reshape(-1, 3)  # Reshape for denormalization
    denormalized = scaler.inverse_transform(reshaped).reshape(image.shape)
    return denormalized.astype(np.uint8)

# Function to perform image clustering
def image_clustering(image_path, n_clusters):
    if not image_path:
        print("No image selected.")
        return

    print("Loading image...")
    image = imread(image_path)
    folder_path, image_file = os.path.split(image_path)

    print("Normalizing image...")
    normalized_image, scaler = normalize_image(image)
    pixels = normalized_image.reshape(-1, 3)

    print(f"Applying K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    clustered_pixels = kmeans.cluster_centers_[labels].reshape(image.shape)

    print("Denormalizing clustered image...")
    clustered_image = denormalize_image(clustered_pixels, scaler)

    clustered_path = os.path.join(folder_path, f"clustered_{image_file}")
    imsave(clustered_path, clustered_image)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(clustered_image)
    plt.title(f"Clustered Image (k={n_clusters})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Clustered image saved at: {clustered_path}")

# Main function
def main():
    dataset_directory = select_dataset_directory()
    if not dataset_directory:
        print("No dataset directory selected.")
        return

    image_path = select_image_file(dataset_directory)
    if not image_path:
        print("No image file selected.")
        return

    n_clusters = get_number_of_clusters()
    if not n_clusters:
        print("Invalid number of clusters selected.")
        return

    image_clustering(image_path, n_clusters)

if __name__ == "__main__":
    main()
