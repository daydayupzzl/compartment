import numpy as np
import multiprocessing
import subprocess
import argparse
import os
import pandas as pd
import cooler
import scipy.sparse
import pickle
import pathlib
import xarray as xr
import traceback
import warnings
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from tools import *

try:
    from scipy.stats import PearsonRConstantInputWarning, SpearmanRConstantInputWarning
except:
    from scipy.stats import ConstantInputWarning as PearsonRConstantInputWarning

cpu_num = multiprocessing.cpu_count()

def aggregate_compartment(cell_table, temp_dir, bins, double_save, output_path):
    comp_pca1_all = []
    comp_pca2_all = []

    corr_pca1_all = []
    corr_pca2_all = []

    stre_pca1_all = []
    stre_pca2_all = []

    best_index_all = []

    for cell_id, cell_url in tqdm(cell_table.items(), desc="Saving cells"):
        if not double_save:
            comp_path = os.path.join(temp_dir, f'{cell_id}.comp.npz')
            npz_data = np.load(comp_path, allow_pickle=True)
            flat_comp = np.concatenate(npz_data['arr_0'])
            comp_pca1_all.append(flat_comp)
        else:
            comp_path = os.path.join(temp_dir, f'{cell_id}.comp.npz')
            npz_data = np.load(comp_path, allow_pickle=True)
            pca1 = npz_data['pca1']
            pca2 = npz_data['pca2']
            best_index = npz_data['best_index'] if 'best_index' in npz_data else None

            # compartment flatten per cell (per bin)
            comp1 = np.concatenate([chrom_res['compartment'] for chrom_res in pca1])
            comp2 = np.concatenate([chrom_res['compartment'] for chrom_res in pca2])

            # per-chromosome scalars
            corr1 = np.array([chrom_res['correlation'] for chrom_res in pca1])
            corr2 = np.array([chrom_res['correlation'] for chrom_res in pca2])
            stre1 = np.array([chrom_res['strength'] for chrom_res in pca1])
            stre2 = np.array([chrom_res['strength'] for chrom_res in pca2])

            comp_pca1_all.append(comp1)
            comp_pca2_all.append(comp2)

            corr_pca1_all.append(corr1)
            corr_pca2_all.append(corr2)
            stre_pca1_all.append(stre1)
            stre_pca2_all.append(stre2)

            if best_index is not None:
                best_index_all.append(np.array(best_index, dtype=int))

    data_vars = {}
    coords = {
        'cell': np.array(cell_table.index),
        'bin': np.array(bins.index)
    }

    chroms = bins['chrom'].drop_duplicates().values
    coords['chrom'] = chroms


    if not double_save:
        data_vars['compartment'] = (['cell', 'bin'], np.vstack(comp_pca1_all))
    else:
        data_vars['compartment_pca1'] = (['cell', 'bin'], np.vstack(comp_pca1_all))
        data_vars['compartment_pca2'] = (['cell', 'bin'], np.vstack(comp_pca2_all))

        data_vars['correlation_pca1'] = (['cell', 'chrom'], np.vstack(corr_pca1_all))
        data_vars['correlation_pca2'] = (['cell', 'chrom'], np.vstack(corr_pca2_all))
        data_vars['strength_pca1'] = (['cell', 'chrom'], np.vstack(stre_pca1_all))
        data_vars['strength_pca2'] = (['cell', 'chrom'], np.vstack(stre_pca2_all))

        if len(best_index_all) > 0:
            n_chrom = best_index_all[0].shape[0]
            for arr in best_index_all:
                if arr.shape[0] != n_chrom:
                    raise ValueError("best_index length mismatch between cells")
            data_vars['best_index'] = (['cell', 'chrom'], np.vstack(best_index_all))

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add bin-level coordinates
    ds.coords['bin_chrom'] = ('bin', np.array(bins['chrom']))
    ds.coords['bin_start'] = ('bin', np.array(bins['start']))
    ds.coords['bin_end'] = ('bin', np.array(bins['end']))

    ds.to_netcdf(output_path)
    return

def choose_PCA_component(E, pca_value):
    r = []
    for i in range(2):
        labels, groups = pd.qcut(pca_value[:, i], 50, labels=False, retbins=True, duplicates='drop')

        n_bins = len(np.unique(labels))

        sad = np.array([[E[np.ix_(labels == x, labels == y)].sum() for x in range(n_bins)] for y in range(n_bins)])
        count = np.array([[(labels == x).sum() * (labels == y).sum() for x in range(n_bins)] for y in range(n_bins)])
        count_nonzero = np.where(count == 0, 1, count)
        sad = sad / count_nonzero

        if n_bins < 10:
            print(f"Warning: PCA-{i + 1} only has {n_bins} bins, skipping saddle strength calculation.")
            r.append(np.nan)
            continue

        top_left = sad[:10, :10].sum()
        bottom_right = sad[-10:, -10:].sum()
        top_right = sad[:10, -10:].sum()
        bottom_left = sad[-10:, :10].sum()

        r.append((top_left + bottom_right) / (top_right + bottom_left + 1e-8))

    if np.isnan(r[0]) or np.isnan(r[1]):
        print("Warning: One or both of the saddle strengths are NaN. Returning default value.")
        return 0, 0, 0

    if r[0] > r[1]:
        i = 0  # PCA-1
    else:
        i = 1  # PCA-2

    return i, r[0], r[1]




def single_cell_compartment(cell_url, cell_index, cpg_profile, output_prefix, chrom_sizes, mode, norm_type, exp_type,
                            double_save, resolution, model=None):

    if mode == 'cool':
        cool = cooler.Cooler(cell_url)
        chroms = pd.Index(cool.chromnames).intersection(cpg_profile['chrom'])
    elif mode == 'npy':
        chroms = chrom_sizes.index

    all_comp = []
    all_comp_pca1 = []
    all_comp_pca2 = []
    best_index_list = []

    chroms = chroms[~chroms.isin(['chrX', 'chrY'])]
    for chrom in chroms:
        cpg_ratio = cpg_profile.loc[cpg_profile['chrom'] == chrom, 'cpg_ratio']

        if mode == 'cool':
            matrix = cool.matrix(balance=False).fetch(chrom)
        elif mode == 'npy':
            if 'bulk' in output_prefix:
                base = os.path.dirname(cell_url)
                data = np.load(os.path.join(base, f"{chrom}_sparse_adj.npy"), allow_pickle=True)
            else: # single-cell calling
                base = os.path.dirname(os.path.dirname(cell_url))
                data = np.load(os.path.join(base, 'raw', f"{chrom}_sparse_adj.npy"), allow_pickle=True)
            matrix = data[cell_index]

        if scipy.sparse.issparse(matrix):
            matrix = matrix.toarray()

        if resolution == 500000:
            matrix = merge_hic_matrix(matrix, original_bin_size=100000, target_bin_size=500000, merge_method='mean')

        if cpg_ratio.shape[0] != matrix.shape[0]:
            print(f'ERROR : cpg_ratio and chrom_size have different shapes at {chrom}', cpg_ratio.shape[0], matrix.shape[0])
            return (0)

        matrix = matrix - np.diag(np.diag(matrix))
        rowsum = matrix.sum(axis=0)
        valid_rows = rowsum > 0
        if np.any(valid_rows):
            q50 = np.percentile(rowsum[valid_rows], 50)
            q99 = np.percentile(rowsum[valid_rows], 99)
            final_thresh = q50 * 2 - q99
            binfilter = rowsum > final_thresh
        else:
            binfilter = np.ones(matrix.shape[0], dtype=bool)
        effective_rows = binfilter

        matrix = matrix[effective_rows][:, effective_rows]

        np.fill_diagonal(matrix, 1)

        normalized_matrix = normalization_metrix(matrix, norm_type)
        oe_matrix = oe(normalized_matrix, exp_type)
        np.fill_diagonal(oe_matrix, 1)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=PearsonRConstantInputWarning
            )
            pearson_matrix = np.corrcoef(oe_matrix)
        np.fill_diagonal(pearson_matrix, 1)
        pearson_matrix[np.isnan(pearson_matrix)] = 0.0

        try:

            if model is not None:
                pca_values = model.transform(pearson_matrix)
            else:
                try:
                    pca = PCA(n_components=2)
                    pca_values = pca.fit_transform(pearson_matrix)
                except Exception as e:
                    print(f"[SKIP] {chrom} PCA SVD error → set all 0", e)
                    all_comp.append(np.zeros(len(effective_rows)))
                    all_comp_pca1.append({
                        'strength': 0,
                        'correlation': 0,
                        'compartment': np.zeros(len(effective_rows))
                    })
                    all_comp_pca2.append({
                        'strength': 0,
                        'correlation': 0,
                        'compartment': np.zeros(len(effective_rows))
                    })
                    continue

            original_n_bins = len(effective_rows)
            pc1_full = np.zeros(original_n_bins)
            pc2_full = np.zeros(original_n_bins)

            pc1_full[effective_rows] = pca_values[:, 0]
            pc2_full[effective_rows] = pca_values[:, 1]

            pca_full = np.column_stack([pc1_full, pc2_full])

            best_index, strength_pca1, strength_pca2 = choose_PCA_component(oe_matrix, pca_values)

            best_index_list.append(best_index + 1)

            pca_value, coor1, coor2 = cpg_calib(cpg_ratio, pca_full, best_index)

            all_comp.append(pca_value)

            all_comp_pca1.append({
                'strength': strength_pca1,
                'correlation': coor1,
                'compartment': pca_full[:, 0]
            })
            all_comp_pca2.append({
                'strength': strength_pca2,
                'correlation': coor2,
                'compartment': pca_full[:, 1]
            })

        except Exception as e:
            error_msg = f"[CRITICAL] PCA failed on {output_prefix} {chrom}, abort this cell completely! Error: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

    if double_save:
        np.savez(f'{output_prefix}.comp.npz',
                 pca1=np.array(all_comp_pca1, dtype=object),
                 pca2=np.array(all_comp_pca2, dtype=object),
                 best_index=np.array(best_index_list, dtype=int)
                 )
    else:
        np.savez(f'{output_prefix}.comp.npz', np.array(all_comp, dtype=object))

def run_single_cell(cell_index, cell_id, cell_url, cpg_profile, temp_dir, chrom_sizes, mode, norm_type, exp_type,
                    double_save, resolution):
    try:
        cell_prefix = f'{temp_dir}/{cell_id}'
        if pathlib.Path(f'{cell_prefix}.comp.npz').exists():
            # print(f'{cell_id} already exists. Skipping.')
            return
        single_cell_compartment(cell_url=cell_url,
                                cell_index=cell_index,
                                cpg_profile=cpg_profile,
                                output_prefix=cell_prefix,
                                chrom_sizes=chrom_sizes,
                                norm_type=norm_type,
                                exp_type=exp_type,
                                double_save=double_save,
                                resolution=resolution,
                                mode=mode)
        print(f'{cell_id} finished.')
    except Exception as e:
        print(f"[CRITICAL] Cell failed completely: {e}")
        traceback.print_exc()
        return None

def multiple_cell_compartment(cell_table_path, output_prefix, cpg_profile_path, dataset, cell_type, chrom_size_path,
                              resolution, norm_type,
                              exp_type, double_save, cpu=10,
                              mode='cool'):
    cell_table = pd.read_csv(cell_table_path, sep='\t', index_col=0, header=None).squeeze(axis=1)
    print(cell_table.shape[0], 'cells to calculate.')

    if cell_type != 'all':
        label_info_path = f'data/Higashi/temp/{dataset}/100K_cool/label_info.pickle'
        with open(label_info_path, 'rb') as file:
            label_info = pickle.load(file)
        if 'cell type' not in label_info.keys():
            raise ValueError("label_info not contains 'cell type' key，please check pickle file！")
        cell_type_series = pd.Series(label_info['cell type'])
        type_indices = cell_type_series[cell_type_series == cell_type].index
        cell_table = cell_table.iloc[type_indices]
        print(f" {cell_type} cell num after filtering：{cell_table.shape[0]} cells to calculate.")

    temp_dir = pathlib.Path(f'{output_prefix}_compartment_temp')
    temp_dir.mkdir(exist_ok=True)
    cpg_profile = pd.read_hdf(cpg_profile_path)

    if mode == 'npy':
        chrom_sizes = pd.read_csv(chrom_size_path, sep='\t', index_col=0, header=None).squeeze(axis=1)
        chrom_sizes = chrom_sizes[~chrom_sizes.index.str.contains('chrx|chry', case=False)]
        bins = cooler.binnify(chrom_sizes, resolution)
    elif mode == 'cool':
        chrom_sizes = None
        bins = cpg_profile[['chrom', 'start', 'end']].copy()
        bins = bins[~bins['chrom'].str.contains('chrX|chrY|chrx|chry', case=False)]
        bins = bins.reset_index(drop=True)
    else:
        print('ERROR : mode need to be cool or npy')
        return

    with tqdm_joblib(tqdm(desc="Processing cells", total=len(cell_table))) as progress_bar:
        Parallel(n_jobs=cpu)(
            delayed(run_single_cell)(cell_index, cell_id, cell_url, cpg_profile, temp_dir, chrom_sizes, mode, norm_type,
                                     exp_type, double_save, resolution)
            for cell_index, (cell_id, cell_url) in enumerate(cell_table.items())
        )

    print("Saving results...")
    output_path = f'{output_prefix}.compartment.nc' if cell_type == 'all' else f'{output_prefix}.{cell_type.replace("/", "_")}.compartment.nc'
    aggregate_compartment(cell_table=cell_table,
                          temp_dir=temp_dir,
                          bins=bins,
                          double_save=double_save,
                          output_path=output_path)

    subprocess.run(f'rm -rf {temp_dir}', shell=True)
    return


def main():
    parser = argparse.ArgumentParser(description='Calculate PCA-based compartment signals')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--cell_table_path', required=True, help='Path to cell table')
    parser.add_argument('--output_prefix', required=True, help='Output file prefix')
    parser.add_argument('--resolution', type=int, required=True, help='resolution to raw data')
    parser.add_argument('--cpu', type=int, default=10, help='Number of CPUs to use')
    parser.add_argument('--mode', choices=['cool', 'npy'], default='cool', help='Data format mode')
    parser.add_argument('--norm_type', choices=['sqrtvc', 'vc', 'none'], default='sqrtvc',help='Normalization method')
    parser.add_argument('--exp_type', choices=['dicrete', 'loess'], default='dicrete', help='Expectation method')
    parser.add_argument('--double_save', type=bool, default=False, help='save PCA1&PCA2')
    parser.add_argument('--cell_type', type=str, default='all', help='cell type')

    args = parser.parse_args()

    if args.dataset not in ['Tian', 'Lee']:
        cpg_profile_path = "files/mm10.fa/cpg_ratio_100k.hdf" if args.resolution == 100000 else "files/mm10.fa/cpg_ratio_500k.hdf"
        chrom_size_path = "files/chromsize/mm10.chrom.sizes"
    else:
        cpg_profile_path = "files/hg38.fa/cpg_ratio_100k.hdf" if args.resolution == 100000 else "files/hg38.fa/cpg_ratio_500k.hdf"
        chrom_size_path = "files/chromsize/hg38.autosomal.chrom.sizes"

    multiple_cell_compartment(
        cell_table_path=args.cell_table_path,
        output_prefix=args.output_prefix,
        cpg_profile_path=cpg_profile_path,
        chrom_size_path=chrom_size_path,
        norm_type=args.norm_type,
        exp_type=args.exp_type,
        cpu=args.cpu,
        dataset=args.dataset,
        cell_type=args.cell_type,
        mode=args.mode,
        double_save=args.double_save,
        resolution=args.resolution,
    )

    print('done')


if __name__ == '__main__':
    main()
