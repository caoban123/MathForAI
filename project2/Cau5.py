import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

class ScratchPCA:
    def __init__(self, n_components = 0.95):
        self.eigen_values = None
        self.eigen_vectors = None
        self.mean_ = None
        self.std = None 
        self.components = None
        self.n_components = n_components
        self.explained_variance_ratio_ = None
    def fit(self, X):
        X = np.array(X).astype(float)
        
        self.mean_ = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0 
        
        X_scaled = (X - self.mean_) / self.std

        n_samples = X_scaled.shape[0]
        cov_matrix = (X_scaled.T.dot(X_scaled)) / (n_samples - 1)

        self.eigen_values, self.eigen_vectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(self.eigen_values)[::-1]
        self.eigen_values = self.eigen_values[idx]
        self.eigen_vectors = self.eigen_vectors[:,idx]

        total_var = np.sum(self.eigen_values)
        self.explained_variance_ratio_ = self.eigen_values / total_var

        if self.n_components < 1:
            cumsum = np.cumsum(self.explained_variance_ratio_)
            k = np.argmax(cumsum >= self.n_components) + 1
        else:
            k = int(self.n_components)
            
        self.components = self.eigen_vectors[:, :k]
        return self

    def transform(self, X):
        X = np.array(X).astype(float)
        X_scaled = (X - self.mean_) / self.std
        X_pca = np.dot(X_scaled, self.components)
        return X_pca
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, X_pca):
        X_rescaled = np.dot(X_pca, self.components.T)
        X_reconstructed = (X_rescaled * self.std) + self.mean_
        return X_reconstructed

def calculate_reconstruction_error(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    return mse

def plot_compression_results(file_name, original_img, results_dict, k_values):
    n_plots = len(k_values) + 1
    plt.figure(figsize=(22, 6)) 
    
    plt.subplot(1, n_plots, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    for i, k in enumerate(k_values):
        plt.subplot(1, n_plots, i + 2)
        recon_img = results_dict[file_name][k]
        
        error = calculate_reconstruction_error(original_img, recon_img)
        
        plt.imshow(recon_img, cmap='gray')
        plt.title(f"k = {k}\nMSE = {error:.2f}") 
        plt.axis('off')
    
    plt.suptitle(f"PCA Compression Results for: {file_name.upper()}", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(f"result_{file_name}.png") 
    plt.show()
if __name__=="__main__":
    file_names = ['airplane', 'butterfly', 'chair', 'crab', 'crocodile']
    k_values = [2, 5, 10, 20, 50, 100]
    results = {}
    original_images = {}

    for file in file_names:
        path = f"image/{file}.png"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Không tìm thấy file: {path}")
            continue
        print(f"Kích thước ảnh {file} là", image.shape)
        img_float = image.astype(float)
        original_images[file] = img_float
        results[file] = {}
        for k in k_values:
            pca = ScratchPCA(n_components=k)
            X_compressed = pca.fit_transform(img_float)
            # Tái tạo ảnh 
            img_recon = pca.inverse_transform(X_compressed)
            results[file][k] = img_recon
    for file in file_names:
        plot_compression_results(file, original_images[file], results, k_values)