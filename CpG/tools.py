import os
import pickle
import numpy as np
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score, v_measure_score

def clustering_metrics(emb, true_labels, n_clusters=None):
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb)
    pred_labels = kmeans.labels_

    if len(set(pred_labels)) > 1:
        sil_score = silhouette_score(emb, pred_labels, metric='cosine')
        ch_score = calinski_harabasz_score(emb, pred_labels)
    else:
        sil_score = None
        ch_score = None

    ari = adjusted_rand_score(true_labels, pred_labels)
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    v = v_measure_score(true_labels, pred_labels)

    return sil_score, ch_score, ari, ami, v

def vis_embedding(embedding, cell_type, batch=None, neuron_subtype=None):
    vec = UMAP(n_components=2, n_neighbors=15, random_state=0).fit_transform(embedding)
    fig = plt.figure(figsize=(15, 10))

    ax = plt.subplot(2, 2, 1)
    sns.scatterplot(x=vec[:, 0], y=vec[:, 1], hue=cell_type, ax=ax, s=10, alpha=0.8, linewidth=0)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1)
    ax.grid(False)

    plt.tight_layout()
    plt.show()

def merge_hic_matrix(matrix, original_bin_size=100000, target_bin_size=500000, merge_method='sum',
                     resize_strategy='pad'):
    factor = target_bin_size // original_bin_size
    if factor < 1:
        raise ValueError("Target bin size must be larger than original bin size.")

    matrix_size = matrix.shape[0]
    remainder = matrix_size % factor

    if remainder != 0:
        if resize_strategy == 'cut':
            cut_size = matrix_size - remainder
            matrix = matrix[:cut_size, :cut_size]
            new_size = cut_size // factor

        elif resize_strategy == 'pad':
            padded_size = matrix_size + (factor - remainder)

            padded_matrix = np.zeros((padded_size, padded_size), dtype=matrix.dtype)

            padded_matrix[:matrix_size, :matrix_size] = matrix
            matrix = padded_matrix
            new_size = padded_size // factor

        else:
            raise ValueError("Invalid resize strategy. Choose from 'cut' or 'pad'.")
    else:
        new_size = matrix_size // factor

    merged_matrix = np.zeros((new_size, new_size))

    for i in range(new_size):
        for j in range(new_size):
            sub_matrix = matrix[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]

            if merge_method == 'sum':
                merged_matrix[i, j] = np.sum(sub_matrix)
            elif merge_method == 'mean':
                merged_matrix[i, j] = np.mean(sub_matrix)
            elif merge_method == 'max':
                merged_matrix[i, j] = np.max(sub_matrix)
            else:
                raise ValueError("Invalid merge method. Choose from 'sum', 'mean', or 'max'.")

    return normalize_matrix(merged_matrix)

def normalize_matrix(matrix):
    matrix = np.array(matrix, dtype=np.float64)
    row_sums = np.sum(matrix, axis=1)
    row_sums[row_sums == 0] = 1
    norm_factors = np.sqrt(row_sums)
    normalized_matrix = matrix / np.outer(norm_factors, norm_factors)
    return normalized_matrix

def get_labels(base_path):
    with open(os.path.join(base_path, "label_info.pickle"), "rb") as f:
        label_info = pickle.load(f)
    return label_info
