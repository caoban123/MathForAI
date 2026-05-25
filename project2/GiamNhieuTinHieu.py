from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
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
# 1. Tạo tín hiệu hình sin giả lập bị nhiễu
t = np.linspace(0, 10, 100)
signal = np.sin(t) 
noise = np.random.normal(0, 0.4, (50, 100)) # 50 bản ghi tín hiệu bị nhiễu
noisy_signal = signal + noise

# 2. Dùng ScratchPCA để lọc
# Giả sử chỉ giữ lại 1 thành phần chính duy nhất (cấu trúc hình sin)
pca_signal = ScratchPCA(n_components=1)
X_pca = pca_signal.fit_transform(noisy_signal)
X_denoised = pca_signal.inverse_transform(X_pca)

# 3. Vẽ so sánh
plt.figure(figsize=(10, 5))
plt.plot(t, noisy_signal[0], alpha=0.3, label="Tín hiệu nhiễu")
plt.plot(t, signal, 'g', label="Tín hiệu gốc", linewidth=2)
plt.plot(t, X_denoised[0], 'r--', label="Sau khi lọc bằng PCA", linewidth=2)
plt.legend()
plt.title("Giảm nhiễu tín hiệu bằng PCA")
plt.grid(True)
plt.savefig("giam_nhieu_tin_hieu_pca.png", dpi=300, bbox_inches="tight")
plt.show()