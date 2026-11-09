import os
import re
import pickle
import argparse
import warnings
from tqdm import tqdm
from tools import *

ROOT_HIGASHI = 'data/Higashi/temp'
ROOT_SCHICLUSTER = 'data/schicluster/data'
ROOT_OUT = 'data/schicluster/all_data'
DATA_TYPES = ['raw', 'schicluster', 'higashi']
HUMAN_DATASETS = ['Lee', 'Tian']


def methods_for(res):
    return ['pca', 'cpg', 'higashi', 'scghost'] if res == 500000 else ['pca', 'cpg', 'higashi']


def _sfx(res):
    if res == 100000:
        return '', ''
    if res == 500000:
        return '_500K', '.500k'
    raise ValueError(f"unsupported resolution {res}")


def bulk_paths(dataset, num, dtype, res):
    h5, nc = _sfx(res)
    if dtype == 'higashi':
        base = f'{ROOT_HIGASHI}/{dataset}/100K_bulk_num/exp1_nbr_5_impute/{num}'
        hdf5 = f'scCompartment_bulk{num}{h5}.hdf5'
    elif dtype == 'schicluster':
        base = f'{ROOT_SCHICLUSTER}/{dataset}/100K_bulk_num/{num}'
        hdf5 = f'scCompartment_bulk{num}_scHiCluster{h5}.hdf5'
    elif dtype == 'raw':
        base = f'{ROOT_HIGASHI}/{dataset}/100K_bulk_num/raw/{num}'
        hdf5 = f'scCompartment_bulk{num}_raw{h5}.hdf5'
    else:
        raise ValueError(dtype)
    pca = f'{dataset}.{dtype}.pca.bulk{num}{nc}.compartment.nc'
    cpg = f'{dataset}.{dtype}.cpg.bulk{num}{nc}.compartment.nc'
    return base, os.path.join(f'{ROOT_HIGASHI}/{dataset}/100K_bulk_num/', hdf5), pca, cpg


def sc_paths(dataset, dtype, res):
    h5, nc = _sfx(res)
    cool = f'{ROOT_HIGASHI}/{dataset}/100K_cool'
    if dtype == 'raw':
        base, hdf5, scg = f'{cool}/raw/', f'{cool}/scCompartment_raw{h5}.hdf5', f'{cool}/scghost_raw/bed_files/'
    elif dtype == 'schicluster':
        base = f'{ROOT_SCHICLUSTER}/{dataset}/100K/'
        hdf5, scg = f'{cool}/scCompartment_scHiCluster{h5}.hdf5', f'{cool}/scghost_schicluster/bed_files/'
    elif dtype == 'higashi':
        base, hdf5, scg = f'{cool}/exp1_nbr_5_impute/', f'{cool}/scCompartment{h5}.hdf5', f'{cool}/scghost/bed_files/'
    else:
        raise ValueError(dtype)
    return (base, hdf5, f'{dataset}.{dtype}.pca{nc}.compartment.nc', f'{dataset}.{dtype}.cpg{nc}.compartment.nc', scg)


def output_pkl(dataset, cell_type, res):
    safe = cell_type.replace('/', '_').replace('\\', '_')
    return f"{ROOT_OUT}/{dataset}_{safe}{'_500k' if res == 500000 else ''}.pkl"


def merge_nums_for(total):
    # bulk-all is always needed as ground truth
    return list(range(100, total + 1, 100)) + ['all']


def bulk_cell_index(bulk_dir, cell_type, merge_num, dtype):
    cell_type = re.sub(r'[\\/:*?"<>|]', '_', cell_type)
    names = pd.read_csv(os.path.join(bulk_dir, 'cell_table_100K.tsv'), sep='\t', header=None).iloc[:, 0].tolist()
    return names.index(cell_type if dtype == 'raw' else f'{cell_type}_bulk_num_{merge_num}')


def genome_info(dataset):
    if dataset in HUMAN_DATASETS:
        return 'files/chromsize/hg38.autosomal.chrom.sizes', [f'chr{i}' for i in range(1, 23)]
    return 'files/chromsize/mm10.chrom.sizes', [f'chr{i}' for i in range(1, 20)]


def run_analysis(dataset, target_cell_type, res=100000):
    import cooler
    import xarray as xr

    methods = methods_for(res)
    with open(f"{ROOT_HIGASHI}/{dataset}/100K_cool/label_info.pickle", "rb") as f:
        label_info = pickle.load(f)
    sc_cell_index_list = [i for i, ct in enumerate(label_info["cell type"]) if ct == target_cell_type]
    if not sc_cell_index_list:
        raise ValueError(f"no cells of type {target_cell_type!r} in {dataset}")
    total = len(sc_cell_index_list)
    merge_nums = merge_nums_for(total)
    print('total', total, '| merge_nums', merge_nums, '| methods', methods)

    chrom_size_path, chromosomes = genome_info(dataset)
    chrom_sizes = cooler.read_chromsizes(chrom_size_path, all_names=True).loc[chromosomes]

    # ------------ pseudo-bulk ------------
    bulk_num_call = {m: {t: {} for t in DATA_TYPES} for m in methods}
    raw_bulk_num_pca_call = {}
    for chr_name in tqdm(chromosomes, desc='bulk level'):
        chr_size = chrom_sizes[chr_name]
        for merge_num in merge_nums:
            key = f'{chr_name}_bulk{merge_num}'
            for dtype in DATA_TYPES:
                base, h5, pca, cpg = bulk_paths(dataset, merge_num, dtype, res)
                idx = bulk_cell_index(base, target_cell_type, merge_num, dtype)
                ds_pca = xr.open_dataset(os.path.join(base, pca))
                ds_cpg = xr.open_dataset(os.path.join(base, cpg))
                bulk_num_call['pca'][dtype][key] = get_ab_centered(ds_pca, idx, chr_name, chr_size, res)
                bulk_num_call['cpg'][dtype][key] = get_ab_centered(ds_cpg, idx, chr_name, chr_size, res, is_cpg=True)
                bulk_num_call['higashi'][dtype][key] = get_ab_centered(h5, idx, chr_name, chr_size, res, h5_file=True)
                if 'scghost' in methods:
                    # scGHOST has no bulk mode: average the single-cell calls of the cells in the pseudo-bulk
                    n = total if merge_num == 'all' else int(merge_num)
                    bulk_num_call['scghost'][dtype][key] = get_ab_scghost(
                        sc_paths(dataset, dtype, res)[4], None, chr_name, chr_size, res,
                        bulk_cell_index_list=sc_cell_index_list[:n])
            base, _, pca, _ = bulk_paths(dataset, merge_num, 'raw', res)
            idx = bulk_cell_index(base, target_cell_type, merge_num, 'raw')
            raw_bulk_num_pca_call[key] = get_ab_centered(xr.open_dataset(os.path.join(base, pca)), idx,
                                                         chr_name, chr_size, res)

    # ------------ single cell ------------
    single_cell_call = {m: {t: {} for t in DATA_TYPES} for m in methods}
    for chr_name in tqdm(chromosomes, desc='single cell level'):
        chr_size = chrom_sizes[chr_name]
        for dtype in DATA_TYPES:
            base, h5, pca, cpg, scg = sc_paths(dataset, dtype, res)
            ds_pca = xr.open_dataset(os.path.join(base, pca))
            ds_cpg = xr.open_dataset(os.path.join(base, cpg))
            calls = {m: [] for m in methods}
            for ci in tqdm(sc_cell_index_list, desc=f"cells {chr_name} ({dtype})"):
                calls['pca'].append(get_ab_centered(ds_pca, ci, chr_name, chr_size, res))
                calls['cpg'].append(get_ab_centered(ds_cpg, ci, chr_name, chr_size, res, is_cpg=True))
                calls['higashi'].append(get_ab_centered(h5, ci, chr_name, chr_size, res, h5_file=True))
                if 'scghost' in methods:
                    calls['scghost'].append(get_ab_scghost(scg, ci, chr_name, chr_size, res))
            for m in methods:
                single_cell_call[m][dtype][chr_name] = calls[m]

    filename = output_pkl(dataset, target_cell_type, res)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump({'single_cell_call': single_cell_call, 'bulk_num_call': bulk_num_call,
                     'raw_bulk_num_pca_call': raw_bulk_num_pca_call}, f)
    print(f"Data saved to {filename}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description='Collect A/B vectors of all methods for one dataset and cell type')
    parser.add_argument('dataset', help='dataset name, e.g. Tan')
    parser.add_argument('cell_type', help='representative cell type of the dataset, e.g. Neuron')
    parser.add_argument('--resolution', type=int, default=100000, choices=[100000, 500000], help='resolution')
    args = parser.parse_args()
    run_analysis(args.dataset, args.cell_type, args.resolution)
