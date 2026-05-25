import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ScratchPCA:
    def __init__(self, n_components = 0.95):
        self.eigen_values = None
        self.eigen_vectors = None
        self.mean = None
        self.std = None 
        self.components = None
        self.n_components = n_components
        self.explained_variance_ratio = None
    def fit(self, X):
        X = np.array(X).astype(float)
        
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0 
        
        X_scaled = (X - self.mean) / self.std

        n_samples = X_scaled.shape[0]
        cov_matrix = (X_scaled.T.dot(X_scaled)) / (n_samples - 1)

        self.eigen_values, self.eigen_vectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(self.eigen_values)[::-1]
        self.eigen_values = self.eigen_values[idx]
        self.eigen_vectors = self.eigen_vectors[:,idx]

        total_var = np.sum(self.eigen_values)
        self.explained_variance_ratio = self.eigen_values / total_var

        if self.n_components < 1:
            cumsum = np.cumsum(self.explained_variance_ratio)
            k = np.argmax(cumsum >= self.n_components) + 1
        else:
            k = int(self.n_components)
            
        self.components = self.eigen_vectors[:, :k]
        return self

    def transform(self, X):
        X = np.array(X).astype(float)
        X_scaled = (X - self.mean) / self.std
        X_pca = np.dot(X_scaled, self.components)
        return X_pca
    def fit_transform(self, X):
        return self.fit(X).transform(X)
if __name__=="__main__":
    original_matrix = np.random.randint(1, 100, size=(5, 10))
    print("Ma trận gốc (5x10):")
    print(original_matrix)

    pca = ScratchPCA(n_components=0.95)
    pca.fit(original_matrix)
    matrix_reduced = pca.transform(original_matrix)

    print("Eigenvalues:\n", pca.eigen_values)
    print("\nEigenvectors:\n", pca.eigen_vectors)

    print("\nSố lượng thành phần chính được chọn (k):", pca.components.shape[1])
    print("\nMa trận sau khi giảm chiều:")
    print(matrix_reduced)
    print("Kích thước mới:", matrix_reduced.shape)

    # Tạo một vector mới (chưa xuất hiện trong ma trận ban đầu)
    new_vector = np.array([45, 67, 23, 89, 12, 54, 76, 31, 58, 90])

    # Chiếu vector mới sang không gian PCA
    new_vector_scaled = (new_vector - pca.mean) / pca.std
    new_vector_projected = np.dot(new_vector_scaled, pca.components)

    print("\nVector mới (10D, chưa có trong dữ liệu gốc):")
    print(new_vector)

    print("\nVector sau khi chiếu sang không gian PCA (kD):")
    print(new_vector_projected)