import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "..", "data", "clean")
REPORTS_DIR = os.path.join(BASE_DIR, "..", "..", "reports")

def load_data():
    metadata_train = pd.read_csv(os.path.join(CLEAN_DIR, "metadata_train.csv"))
    features_train = pd.read_csv(os.path.join(CLEAN_DIR, "features_train.csv"))
    return metadata_train, features_train

def plot_k_distance(features, k=5):
    print(f"Calculating K-distance plot (k={k})...")
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    # Sort distances to the k-th neighbor
    k_distances = np.sort(distances[:, k-1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.title(f'K-distance Plot (k={k})')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.grid(True)
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, 'k_distance_plot.png'))
    plt.close()
    print(f"K-distance plot saved to {os.path.join(REPORTS_DIR, 'k_distance_plot.png')}")
    
    return k_distances

def plot_tsne(features, metadata, n_samples=1000):
    print(f"Calculating t-SNE for {n_samples} samples...")
    # Sample data if too large for t-SNE
    if len(features) > n_samples:
        idx = np.random.choice(len(features), n_samples, replace=False)
        features_sample = features.iloc[idx]
        metadata_sample = metadata.iloc[idx]
    else:
        features_sample = features
        metadata_sample = metadata
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    projections = tsne.fit_transform(features_sample)
    
    df_tsne = pd.DataFrame(projections, columns=['x', 'y'])
    df_tsne['genre'] = metadata_sample['genre_group'].values
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_tsne, x='x', y='y', hue='genre', palette='viridis', alpha=0.7)
    plt.title('t-SNE Visualization of Song Features')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(REPORTS_DIR, 'tsne_visualization.png'))
    plt.close()
    print(f"t-SNE visualization saved to {os.path.join(REPORTS_DIR, 'tsne_visualization.png')}")

def calculate_metrics(features, metadata, k_distances):
    metrics = {
        "mean_k_distance": np.mean(k_distances),
        "median_k_distance": np.median(k_distances),
        "std_k_distance": np.std(k_distances),
        "max_k_distance": np.max(k_distances)
    }
    
    print("\n--- Model Quality Metrics ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    n_samples = min(2000, len(features))
    idx = np.random.choice(len(features), n_samples, replace=False)
    features_sample = features.iloc[idx]
    labels_sample = metadata.iloc[idx]['genre_group']

    valid_mask = labels_sample.notna()
    features_sample = features_sample[valid_mask]
    labels_sample = labels_sample[valid_mask]
    
    if len(labels_sample.unique()) > 1:
        sil = silhouette_score(features_sample, labels_sample)
        db_index = davies_bouldin_score(features_sample, labels_sample)
        print(f"Silhouette Score (Genre Group): {sil:.4f}")
        print(f"Davies-Bouldin Index (Genre Group): {db_index:.4f}")
        metrics["silhouette_score"] = sil
        metrics["davies_bouldin_index"] = db_index
    
    return metrics

if __name__ == "__main__":
    metadata_train, features_train = load_data()

    k_distances = plot_k_distance(features_train, k=5)

    plot_tsne(features_train, metadata_train, n_samples=2000)

    calculate_metrics(features_train, metadata_train, k_distances)
