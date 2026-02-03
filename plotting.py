import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

def plot_average_activation_norms(avg_norms):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(avg_norms)), avg_norms, marker='o')
    plt.xlabel("Transformer Block")
    plt.ylabel("Average L2 Norm")
    plt.title("Average Activation Norm per Transformer Block")
    plt.show()

def plot_token_norms(norms, worm_token_indices=None, title="Activation Norms"):
    plt.figure(figsize=(12, 4))
    plt.plot(norms, marker='o', label="All tokens")
    if worm_token_indices is not None:
        # Ensure indices are within bounds
        valid_indices = [i for i in worm_token_indices if i < len(norms)]
        worm_norms = [norms[i] for i in valid_indices]
        plt.scatter(valid_indices, worm_norms, color='red', s=100, label="Worm Tokens")
    plt.title(title)
    plt.xlabel("Token Index")
    plt.ylabel("L2 Norm")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_heatmap(data, title="Heatmap", xlabel="Hidden Dimension", ylabel="Token Index"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, cmap="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def visualize_embeddings(embeddings, labels):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=100)
        plt.text(embeddings_2d[i, 0] + 0.01, embeddings_2d[i, 1] + 0.01, label, fontsize=9)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Embedding Space Visualization")
    plt.show()
