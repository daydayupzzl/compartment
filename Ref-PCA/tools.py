from joblib import Parallel, delayed
from Higashi_backend.utils import *
from Higashi_analysis.Higashi_analysis import *


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


def compute_compartment(temp, use_rows_list, slice_start_list, slice_end_list, bulk_model_list, bulk_reverse_list):
    temp_compartment_list = []
    for j in range(len(use_rows_list)):
        slice_start, slice_end = slice_start_list[j], slice_end_list[j]
        temp_slice = temp[slice_start:slice_end, :]
        temp_slice = temp_slice[:, slice_start:slice_end]
        temp_select = temp_slice[use_rows_list[j], :]
        temp_select = temp_select[:, use_rows_list[j]]

        temp_compartment = compartment(temp_select, False, bulk_model_list[j], None)
        if bulk_reverse_list[j]:
            temp_compartment = -1 * temp_compartment
        temp_compartment_list.append(temp_compartment.reshape((-1)))
    return temp_compartment_list


def process_single_cell(i, matrix_file_path, mask, size, use_rows_list, slice_start_list, slice_end_list,
                        bulk_model_list, bulk_reverse_list, is_500K):
    impute_data = np.load(matrix_file_path, allow_pickle=True)
    matrix = impute_data[i]

    temp = np.zeros((size, size))
    proba = matrix.toarray()
    if is_500K:
        proba = merge_hic_matrix(proba, original_bin_size=100000, target_bin_size=500000, merge_method='mean')
    xs, ys = proba.nonzero()
    temp[xs.astype('int'), ys.astype('int')] += proba[xs.astype('int'), ys.astype('int')]
    temp = temp + temp.T
    temp *= mask

    compartments = compute_compartment(temp, use_rows_list, slice_start_list, slice_end_list,
                                       bulk_model_list, bulk_reverse_list)
    return compartments


def process_cool_file(matrix_file_path, mask, use_rows_list, slice_start_list, slice_end_list,
                      bulk_model_list, bulk_reverse_list, size, chrom, mode, is_500K):
    if mode == 'cool':
        c = cooler.Cooler(matrix_file_path)
        mat = c.matrix(balance=False).fetch(chrom)
        if is_500K:
            mat = merge_hic_matrix(mat, original_bin_size=100000, target_bin_size=500000, merge_method='mean')
        xs, ys = np.nonzero(mat)
        proba = mat[xs, ys]
        temp = np.zeros((size, size))
        temp[xs, ys] += proba
        temp = temp + temp.T
        temp *= mask

        return compute_compartment(temp, use_rows_list, slice_start_list, slice_end_list,
                                   bulk_model_list, bulk_reverse_list)

    elif mode == 'npy':
        impute_data = np.load(matrix_file_path, allow_pickle=True)

        results = Parallel(n_jobs=-1)(delayed(process_single_cell)(
            i, matrix_file_path, mask, size, use_rows_list, slice_start_list, slice_end_list,
            bulk_model_list, bulk_reverse_list, is_500K
        ) for i in trange(len(impute_data)))

        temp_compartment_list_all = [[] for _ in range(len(use_rows_list))]
        for cell_result in results:
            for j in range(len(cell_result)):
                temp_compartment_list_all[j].append(cell_result[j])
        return temp_compartment_list_all

    else:
        print('ERROR : Must provide either Imputation data or raw data')
