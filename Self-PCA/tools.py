import numpy as np
from scipy.stats import spearmanr, pearsonr, zscore

def normalization_metrix(matrix, norm_type):
    if norm_type == 'sqrtvc':
        return sqrtvc_normalization(matrix)
    elif norm_type == 'vc':
        return vc_normalization(matrix)
    elif norm_type == 'none':
        return matrix

def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def cpg_calib(cpgtmp, pcs, i):
    if pearsonr(cpgtmp, pcs[:, i])[0] > 0:
        pc = pcs[:, i]
    else:
        pc = -pcs[:, i]
    return pc, pearsonr(cpgtmp, pcs[:, 0])[0], pearsonr(cpgtmp, pcs[:, 1])[0]

def oe(matrix, exp_type, loess_frac=0.1):
    n = matrix.shape[0]
    new_matrix = np.zeros_like(matrix)

    expected = compute_expected_by_distance(matrix)

    if exp_type == 'loess':
        distances = np.arange(n)
        expected = loess_smooth(distances, expected, frac=loess_frac)
    for k in range(n):
        rows, cols = kth_diag_indices(matrix, k)
        diag = np.diag(matrix, k)
        if expected[k] <= 0:
            new_matrix[rows, cols] = 0.0
        else:
            new_matrix[rows, cols] = diag / expected[k]

    new_matrix = new_matrix + new_matrix.T

    return new_matrix

def vc_normalization(mat, eps=1e-10):
    mat = mat.astype(float)
    coverage = mat.sum(axis=1)
    coverage[coverage < eps] = np.nan

    norm = np.outer(coverage, coverage)
    mat_vc = mat / norm

    mat_vc[np.isnan(mat_vc)] = 0.0
    return mat_vc
def sqrtvc_normalization(mat, eps=1e-10):
    mat = mat.astype(float)
    coverage = mat.sum(axis=1)
    coverage[coverage < eps] = np.nan

    norm = np.sqrt(np.outer(coverage, coverage))
    mat_sqrtvc = mat / norm

    mat_sqrtvc[np.isnan(mat_sqrtvc)] = 0.0
    return mat_sqrtvc


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
    """
    对称归一化 Hi-C 矩阵。
    """
    matrix = np.array(matrix, dtype=np.float64)
    row_sums = np.sum(matrix, axis=1)
    row_sums[row_sums == 0] = 1
    norm_factors = np.sqrt(row_sums)
    normalized_matrix = matrix / np.outer(norm_factors, norm_factors)
    return normalized_matrix

def compute_expected_by_distance(matrix):
    """
    计算每个 genomic distance k 的 expected（未平滑）
    """
    n = matrix.shape[0]
    expected = np.zeros(n)

    for k in range(n):
        diag = np.diag(matrix, k)
        nz = diag[diag != 0.0]
        expected[k] = np.sum(nz) / (len(nz) + 1e-15)

    return expected

def loess_smooth(x, y, frac=0.1):
    n = len(x)
    y_smooth = np.zeros_like(y)

    window = max(2, int(frac * n))

    for i in range(n):
        left = max(0, i - window // 2)
        right = min(n, i + window // 2 + 1)

        x_local = x[left:right]
        y_local = y[left:right]

        dist = np.abs(x_local - x[i])
        max_dist = np.max(dist) + 1e-12
        w = (1 - (dist / max_dist) ** 3) ** 3

        y_smooth[i] = np.sum(w * y_local) / (np.sum(w) + 1e-15)

    return y_smooth
