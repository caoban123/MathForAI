import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
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

def visualize_pca_steps(X_org, y, target_names):
    # Khởi tạo và FIT đối tượng để lưu lại các thuộc tính 
    pca_3d_obj = ScratchPCA(n_components=3).fit(X_org)
    pca_2d_obj = ScratchPCA(n_components=2).fit(X_org)
    pca_1d_obj = ScratchPCA(n_components=1).fit(X_org)
    
    # Lấy dữ liệu đã transform từ các đối tượng trên
    X_2d_transformed = pca_2d_obj.transform(X_org)
    X_1d_transformed = pca_1d_obj.transform(X_org)
    
    fig = plt.figure(figsize=(20, 6))
    colors = ['navy', 'turquoise', 'darkorange']
    pc_colors, pc_labels = ['red', 'green', 'purple'], ['PC1', 'PC2', 'PC3']

    # --- Subplot 1: Dữ liệu gốc 3D & PC Vectors 
    ax1 = fig.add_subplot(131, projection='3d')
    for color, i, name in zip(colors, [0, 1, 2], target_names):
        ax1.scatter(X_org[y == i, 0], X_org[y == i, 1], X_org[y == i, 2], color=color, label=name, alpha=0.5)
    
    for i in range(3):
        start, direction = pca_3d_obj.mean, pca_3d_obj.eigen_vectors[:, i]
        ax1.quiver(start[0], start[1], start[2], direction[0], direction[1], direction[2], 
                   color=pc_colors[i], length=3.0, normalize=True, linewidths=3)
        
        txt_pos = start + direction * 4.2 
        ax1.text(txt_pos[0], txt_pos[1], txt_pos[2], pc_labels[i], color=pc_colors[i], 
                 fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax1.set_title('1. Dữ liệu gốc 3D & PC Vectors')
    ax1.view_init(elev=20, azim=60)
    ax1.legend()

    # --- Subplot 2: Giảm chiều xuống 2D
    ax2 = fig.add_subplot(132)
    for color, i, name in zip(colors, [0, 1, 2], target_names):
        ax2.scatter(X_2d_transformed[y == i, 0], X_2d_transformed[y == i, 1], color=color, label=name, alpha=0.7)
    ax2.set_title('2. Giảm chiều xuống 2D')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(True, alpha=0.3)

    # --- Subplot 3: Giảm chiều xuống 1D
    ax3 = fig.add_subplot(133)
    for color, i, name in zip(colors, [0, 1, 2], target_names):
        ax3.scatter(X_1d_transformed[y == i, 0], np.zeros_like(X_1d_transformed[y == i, 0]), color=color, label=name, alpha=0.7)
    ax3.set_title('3. Giảm chiều xuống 1D')
    ax3.get_yaxis().set_visible(False)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cau3_vis.png')
    plt.show()

    k2 = pca_2d_obj.components.shape[1]
    k1 = pca_1d_obj.components.shape[1]
    
    print("Tỷ lệ phương sai từng thành phần (của dữ liệu 3D):", pca_3d_obj.explained_variance_ratio)
    print(f"Phương sai giữ lại khi dùng 2D: {np.sum(pca_2d_obj.explained_variance_ratio[:k2])*100:.2f}%")
    print(f"Phương sai giữ lại khi dùng 1D: {np.sum(pca_1d_obj.explained_variance_ratio[:k1])*100:.2f}%")


if __name__ == "__main__":
    iris = load_iris()
    visualize_pca_steps(iris.data[:, :3], iris.target, iris.target_names)
