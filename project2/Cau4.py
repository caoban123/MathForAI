import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
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
def visualize_pca_steps(X_data, y, target_names, PCA_class, file_name):
    pca_3d_obj = PCA_class(n_components=3).fit(X_data)
    pca_2d_obj = PCA_class(n_components=2).fit(X_data)
    pca_1d_obj = PCA_class(n_components=1).fit(X_data)
    
    is_sklearn = hasattr(pca_3d_obj, 'components_')
    
    ev_ratio = pca_3d_obj.explained_variance_ratio_ 

    mean_vec = pca_3d_obj.mean_
    if is_sklearn:
        eigen_vectors = pca_3d_obj.components_.T 
    else:
        eigen_vectors = pca_3d_obj.eigen_vectors

    X_2d = pca_2d_obj.transform(X_data)
    X_1d = pca_1d_obj.transform(X_data)
    
    # 4. Vẽ đồ thị
    fig = plt.figure(figsize=(20, 6))
    colors = ['navy', 'turquoise', 'darkorange']
    pc_colors, pc_labels = ['red', 'green', 'purple'], ['PC1', 'PC2', 'PC3']

    # --- Subplot 1: Dữ liệu gốc 3D & PC Vectors ---
    ax1 = fig.add_subplot(131, projection='3d')
    for color, i, name in zip(colors, [0, 1, 2], target_names):
        ax1.scatter(X_data[y == i, 0], X_data[y == i, 1], X_data[y == i, 2], 
                    color=color, label=name, alpha=0.5)
    
    for i in range(3):
        direction = eigen_vectors[:, i]
        ax1.quiver(mean_vec[0], mean_vec[1], mean_vec[2], 
                   direction[0], direction[1], direction[2], 
                   color=pc_colors[i], length=3.0, normalize=True, linewidths=3)
        
        # Đặt nhãn PC không bị đè
        txt_pos = mean_vec + direction * 4.2 
        ax1.text(txt_pos[0], txt_pos[1], txt_pos[2], pc_labels[i], color=pc_colors[i], 
                 fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax1.set_title(f'1. Gốc (3D) & PC - {file_name}')
    ax1.view_init(elev=20, azim=60)
    ax1.legend()

    # --- Subplot 2: Giảm chiều xuống 2D ---
    ax2 = fig.add_subplot(132)
    for color, i, name in zip(colors, [0, 1, 2], target_names):
        ax2.scatter(X_2d[y == i, 0], X_2d[y == i, 1], color=color, label=name, alpha=0.7)
    ax2.set_title(f'2. Giảm xuống 2D ({file_name})')
    ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2'); ax2.grid(True, alpha=0.3)

    # --- Subplot 3: Giảm chiều xuống 1D ---
    ax3 = fig.add_subplot(133)
    for color, i, name in zip(colors, [0, 1, 2], target_names):
        ax3.scatter(X_1d[y == i, 0], np.zeros_like(X_1d[y == i, 0]), color=color, label=name, alpha=0.7)
    ax3.set_title(f'3. Giảm xuống 1D ({file_name})')
    ax3.get_yaxis().set_visible(False); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

    return X_2d, ev_ratio

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    # CHẠY VÀ ĐO THỜI GIAN SCRATCH PCA ---
    start_manual = time.perf_counter()
    my_pca_obj = ScratchPCA(n_components=2)
    X_my_pca = my_pca_obj.fit_transform(X_std)
    end_manual = time.perf_counter()
    time_manual = end_manual - start_manual

    # CHẠY VÀ ĐO THỜI GIAN SKLEARN PCA ---
    start_sklearn = time.perf_counter()
    sk_pca_obj = PCA(n_components=2)
    X_sk_pca = sk_pca_obj.fit_transform(X_std)
    end_sklearn = time.perf_counter()
    time_sklearn = end_sklearn - start_sklearn

    res_my, ev_my = visualize_pca_steps(X_std, y, target_names, ScratchPCA, 'PCA_Thu_Cong.png')
    res_sk, ev_sk = visualize_pca_steps(X_std, y, target_names, PCA, 'PCA_Sklearn.png')


    print(f"Tốc độ xử lý:")
    print(f"   - ScratchPCA: {time_manual:.6f} giây")
    print(f"   - Sklearn PCA: {time_sklearn:.6f} giây")
    print(f"   => Sklearn nhanh hơn gấp {time_manual/time_sklearn:.2f} lần")

  
    ev_my_2 = ev_my[:2]
    ev_sk_2 = ev_sk[:2]
    ev_diff = np.abs(ev_my_2 - ev_sk_2)
    print(f"\nPhương sai giữ lại (Explained Variance Ratio):")
    print(f"   - Thủ công: {ev_my_2}")
    print(f"   - Thư viện: {ev_sk_2}")
    print(f"   - Sai số trung bình: {np.mean(ev_diff):.2e}")

    # dùng trị tuyệt đối để tránh lỗi ngược hướng vector riêng
    mae_coords = np.mean(np.abs(np.abs(X_my_pca) - np.abs(X_sk_pca)))
    print(f"\nSai số tọa độ (MAE):")
    print(f"   - Giá trị: {mae_coords:.2e}")
    print(f"   => Nhận xét: {'Hoàn toàn khớp' if mae_coords < 1e-10 else 'Có sai số nhỏ'}")
    


