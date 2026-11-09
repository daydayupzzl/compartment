import os
import h5py
import numpy as np
import pandas as pd

def sqrt_norm(matrix):
    coverage = (np.sqrt(np.sum(matrix, axis=-1)))
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = matrix / coverage.reshape((-1, 1))
        matrix = matrix / coverage.reshape((1, -1))
    matrix[np.isnan(matrix)] = 0.0
    matrix[np.isinf(matrix)] = 0.0
    return matrix


def oe(matrix, expected=None):
    new_matrix = np.zeros_like(matrix)
    for k in range(len(matrix)):
        rows, cols = kth_diag_indices(matrix, k)
        diag = np.diag(matrix, k)
        if expected is not None:
            expect = expected[k]
        else:
            expect = np.sum(diag) / (np.sum(diag != 0.0) + 1e-15)
        if expect == 0:
            new_matrix[rows, cols] = 0.0
        else:
            new_matrix[rows, cols] = diag / expect
    new_matrix = new_matrix + new_matrix.T
    return new_matrix


def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def get_OE_matrix(Q):
    Q = Q - np.diag(np.diag(Q))
    oe_matrix = oe(Q, None)
    np.fill_diagonal(oe_matrix, 1)
    return oe_matrix

def denoise_ab_vector(ab_vector, min_segment_len=2):
    vec = ab_vector.copy()
    if len(vec) == 0:
        return vec

    sign_vec = np.sign(vec + 1e-9)

    segments = []
    start = 0
    for i in range(1, len(sign_vec)):
        if sign_vec[i] != sign_vec[i - 1]:
            segments.append((start, i - 1))
            start = i
    segments.append((start, len(sign_vec) - 1))

    for i in range(len(segments)):
        s, e = segments[i]
        length = e - s + 1

        if length < min_segment_len:
            target_mean = 0.0

            if i == 0 and len(segments) > 1:
                next_s, next_e = segments[i + 1]
                target_mean = np.mean(vec[next_s: next_e + 1])

            elif i == len(segments) - 1 and i > 0:
                prev_s, prev_e = segments[i - 1]
                target_mean = np.mean(vec[prev_s: prev_e + 1])

            elif 0 < i < len(segments) - 1:
                left_s, left_e = segments[i - 1]
                right_s, right_e = segments[i + 1]

                left_len = left_e - left_s + 1
                right_len = right_e - right_s + 1

                if left_len >= right_len:
                    target_mean = np.mean(vec[left_s: left_e + 1])
                else:
                    target_mean = np.mean(vec[right_s: right_e + 1])

            vec[s: e + 1] = target_mean

    return vec


def find_boundaries(ab_vector):
    vec = ab_vector.copy()
    sign_vec = np.sign(vec)
    sign_diff = np.diff(sign_vec)
    boundaries = np.where(sign_diff != 0)[0] + 1
    return boundaries


def boundary_overlap_metrics(pred_vec, true_vec, denoise=False, tolerance=1):
    if denoise:
        pred_vec = denoise_ab_vector(pred_vec)
        true_vec = denoise_ab_vector(true_vec)

    pred_bound = find_boundaries(pred_vec)
    true_bound = find_boundaries(true_vec)

    # TP / FP / FN
    TP = sum(1 for pb in pred_bound if np.any(np.abs(true_bound - pb) <= tolerance))
    FP = len(pred_bound) - TP
    FN = len(true_bound) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    jaccard = TP / (len(pred_bound) + len(true_bound) - TP) if (len(pred_bound) + len(true_bound) - TP) > 0 else 0

    return precision, recall, f1, jaccard


def compute_saddle(E, ab_value):
    if E.shape[0] != len(ab_value) or E.shape[1] != len(ab_value):
        raise ValueError(f"contact matrix {E.shape} and A/B vector ({len(ab_value)} bins) have different resolutions")
    labels, groups = pd.qcut(ab_value, 50, labels=False, retbins=True, duplicates='drop')
    n_bins = len(np.unique(labels))

    sad = np.array([[E[np.ix_(labels == x, labels == y)].sum() for x in range(n_bins)] for y in range(n_bins)])
    count = np.array([[(labels == x).sum() * (labels == y).sum() for x in range(n_bins)] for y in range(n_bins)])
    count_nonzero = np.where(count == 0, 1, count)
    sad = sad / count_nonzero

    if n_bins < 10:
        print(f"Warning: compartment ab_value only has {n_bins} bins, skipping saddle strength calculation.")

    # qcut label 0 = lowest ab values = B
    BB_mean = sad[:10, :10].mean()
    AA_mean = sad[-10:, -10:].mean()
    BA_mean = sad[:10, -10:].mean()
    AB_mean = sad[-10:, :10].mean()

    BB_sum = sad[:10, :10].sum()
    AA_sum = sad[-10:, -10:].sum()
    BA_sum = sad[:10, -10:].sum()
    AB_sum = sad[-10:, :10].sum()

    saddle_strength = (AA_sum + BB_sum) / (AB_sum + BA_sum + 1e-8)

    return {
        'saddle_matrix': sad,
        'saddle_strength': saddle_strength,
        'AA': AA_mean,
        'BB': BB_mean,
        'AB': AB_mean,
        'BA': BA_mean
    }


def normalize(vec):
    return vec / np.max(np.abs(vec)) if np.max(np.abs(vec)) != 0 else vec


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


def get_ab_centered(ab, cell_index, chr_name, chrom_size, resolution, h5_file=False, merge_500k=False, is_cpg=False):
    total_bins = int(np.ceil(chrom_size / resolution))

    if h5_file:
        with h5py.File(ab, 'r') as f:
            try:
                comp_ab_cell = f[f'compartment/cell_{cell_index}'][:]
            except KeyError:
                print(f'cannot find the cell {cell_index} in {ab}')
                return np.zeros(total_bins, dtype=np.float32)

            chrom_data = f['compartment/bin/chrom'][:]
            bin_start_list = f['compartment/bin/start'][:]

            chr_name_enc = chr_name.encode() if isinstance(chr_name, str) else chr_name

            chr_mask = chrom_data == chr_name_enc
            chr_bin_idx = np.where(chr_mask)[0]

            valid_bin_starts = bin_start_list[chr_bin_idx]
            valid_bin_ids = valid_bin_starts // resolution
            valid_values = comp_ab_cell[chr_bin_idx]

            full_comp_ab = np.full(total_bins, np.nan, dtype=np.float32)
            valid_mask = valid_bin_ids < total_bins
            valid_bin_ids = valid_bin_ids[valid_mask]
            valid_values = valid_values[valid_mask]
            full_comp_ab[valid_bin_ids] = valid_values

            valid_mean = np.nanmean(full_comp_ab)
            centered_ab_chrm = full_comp_ab - valid_mean
            centered_ab_chrm = np.nan_to_num(centered_ab_chrm, nan=0.0)

    else:
        chr_bins_mask = ab['bin_chrom'].values == chr_name
        comp_ab_cell = ab['compartment'].isel(cell=cell_index)
        comp_ab_chrm = comp_ab_cell.sel(bin=chr_bins_mask).values

        if len(comp_ab_chrm) > total_bins:
            comp_ab_chrm = comp_ab_chrm[:total_bins]
        elif len(comp_ab_chrm) < total_bins:
            padding = np.zeros(total_bins - len(comp_ab_chrm))
            comp_ab_chrm = np.concatenate([comp_ab_chrm, padding])

        if is_cpg:
            centered_ab_chrm = comp_ab_chrm - np.mean(comp_ab_chrm)
        else:
            centered_ab_chrm = comp_ab_chrm

    if merge_500k:
        n = len(centered_ab_chrm)
        if n >= 5:
            trim_len = (n // 5) * 5
            trimmed = centered_ab_chrm[:trim_len]
            centered_ab_chrm = trimmed.reshape(-1, 5).mean(axis=1)

    normalized_ab_chrm = normalize(centered_ab_chrm.reshape(1, -1))[0]
    return normalized_ab_chrm


SCGHOST_SUBCOMP_VALUE = {'scA1': 1, 'scA2': 1, 'scB1': -1, 'scB2': -1, 'scB3': -0.5}


def _scghost_cell_bins(args):
    bed_path, chr_name, resolution, total_bins, subcomp_map = args
    bin_sum = np.zeros(total_bins, dtype=np.float32)
    bin_count = np.zeros(total_bins, dtype=np.float32)
    df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[0, 1, 3], names=['chr', 'start', 'subcompartment'])
    chr_df = df[df['chr'] == chr_name]
    if not chr_df.empty:
        bin_ids = (chr_df['start'] // resolution).astype(int)
        values = chr_df['subcompartment'].map(subcomp_map)
        valid = values.notna()
        np.add.at(bin_sum, bin_ids[valid].to_numpy(), values[valid].to_numpy())
        np.add.at(bin_count, bin_ids[valid].to_numpy(), 1)
    return bin_sum, bin_count


def get_ab_scghost(bed_file_directory, cell_index, chr_name, chr_size, resolution, bulk_cell_index_list=None,
                   n_jobs=4):
    total_bins = int(np.ceil(chr_size / resolution))
    indices = bulk_cell_index_list if bulk_cell_index_list is not None else [cell_index]
    tasks = [(os.path.join(bed_file_directory, f'cell_{idx}.bed'), chr_name, resolution, total_bins,
              SCGHOST_SUBCOMP_VALUE) for idx in indices]
    if n_jobs == 1 or len(tasks) == 1:
        results = [_scghost_cell_bins(t) for t in tasks]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(_scghost_cell_bins)(t) for t in tasks)
    bin_sum = np.sum([r[0] for r in results], axis=0)
    bin_count = np.sum([r[1] for r in results], axis=0)

    mean_vals = np.full(total_bins, np.nan, dtype=np.float32)
    valid = bin_count > 0
    mean_vals[valid] = bin_sum[valid] / bin_count[valid]
    if not valid.any():
        import warnings
        warnings.warn(f"no scGHOST calls on {chr_name} in {bed_file_directory} for {len(indices)} cell(s); "
                      f"returning an all-zero vector")
        return np.zeros(total_bins, dtype=np.float32)
    centered = np.nan_to_num(mean_vals - np.nanmean(mean_vals), nan=0.0)
    return normalize(centered.reshape(1, -1))[0]

