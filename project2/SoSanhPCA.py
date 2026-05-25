import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
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

if __name__ == "__main__":
    # 1. Chuẩn bị dữ liệu
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names
    y_named = [target_names[i] for i in y]
    X_std = StandardScaler().fit_transform(X)

    pca_res = PCA(n_components=2).fit_transform(X_std)
    
    tsne_res = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_std)
    
    lda_res = LDA(n_components=2).fit_transform(X_std, y)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
    my_palette = {"setosa": "navy", "versicolor": "turquoise", "virginica": "darkorange"}

    # --- Subplot 1: PCA ---
    sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=y_named, 
                    palette=my_palette, ax=ax1, s=100, edgecolor='w', alpha=0.8)
    ax1.set_title("PCA\n(Tuyến tính - Không giám sát)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    # --- Subplot 2: t-SNE ---
    sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], hue=y_named, 
                    palette=my_palette, ax=ax2, s=100, edgecolor='w', alpha=0.8)
    ax2.set_title("t-SNE\n(Phi tuyến - Không giám sát)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("t-SNE Dim 1")
    ax2.set_ylabel("t-SNE Dim 2")

    # --- Subplot 3: LDA ---
    sns.scatterplot(x=lda_res[:, 0], y=lda_res[:, 1], hue=y_named, 
                    palette=my_palette, ax=ax3, s=100, edgecolor='w', alpha=0.8)
    ax3.set_title("LDA\n(Tuyến tính - Có giám sát)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("LD1")
    ax3.set_ylabel("LD2")

    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.get_legend().remove()

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12, 
               title="Loài hoa (Iris Species)", title_fontsize=13, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('pca_vs_tsne_vs_lda.png', dpi=300, bbox_inches='tight')
    plt.show()