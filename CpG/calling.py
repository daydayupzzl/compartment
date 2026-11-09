import subprocess
import pandas as pd
from scipy.sparse import diags, csr_matrix
import cooler
import argparse
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import xarray as xr
from tools import *
from tqdm import tqdm

def get_cpg_profile(fasta_path, hdf_output_path, cell_url=None, chrom_size_path=None, resolution=100000):
    temp_bed_path = f'{hdf_output_path}_bins.bed'
    temp_cpg_path = f'{hdf_output_path}_cpg_profile.txt'

    if cell_url is not None:
        cool = cooler.Cooler(cell_url)
        bins = cool.bins()[:]
    elif chrom_size_path is not None:
        chrom_sizes = pd.read_csv(chrom_size_path, sep='\t', index_col=0, header=None).squeeze(axis=1)
        bins = cooler.binnify(chrom_sizes, resolution)
    else:
        print('ERROR : Must provide either cell_url or chrom_size_path')
    bins.to_csv(temp_bed_path, index=None, header=None, sep='\t')
    subprocess.run(
        f'bedtools nuc -fi {fasta_path} -bed {temp_bed_path} -pattern CG -C > {temp_cpg_path}',
        shell=True)

    result = pd.read_csv(temp_cpg_path, sep='\t')
    result['ratio'] = (result['13_user_patt_count'] /
                       (result['12_seq_len'] - result['10_num_N'])).fillna(0)
    result = result.iloc[:, [0, 1, 2, -1]]
    result.columns = ['chrom', 'start', 'end', 'cpg_ratio']
    result.to_hdf(hdf_output_path, key='data')

    # cleanup
    subprocess.run(f'rm -f {temp_bed_path} {temp_cpg_path}', shell=True)
    return


def compartment_strength(matrix, comp, cpg_ratio):
    bin_filter = cpg_ratio > 0

    tmp = comp[bin_filter]
    a_pos = (tmp > np.percentile(tmp, 80))
    b_pos = (tmp < np.percentile(tmp, 20))
    E = matrix.tocoo()
    decay = np.array([E.diagonal(i).mean() for i in range(E.shape[0])])
    E.data = E.data / decay[np.abs(E.col - E.row)]
    E = E.tocsr()[np.ix_(bin_filter, bin_filter)]
    aa = E[np.ix_(a_pos, a_pos)].sum()
    bb = E[np.ix_(b_pos, b_pos)].sum()
    ab = E[np.ix_(a_pos, b_pos)].sum()
    scores = np.array([aa, bb, ab])
    return scores


def single_chrom_compartment(matrix, cpg_ratio, calc_strength=False):
    matrix = matrix - diags(matrix.diagonal())  # remove diag
    # matrix = matrix + matrix.T  # make full matrix
    matrix = matrix + diags((matrix.sum(axis=0).A.ravel() == 0).astype(int))
    matrix.data = matrix.data / np.repeat(matrix.sum(axis=0).A1, matrix.getnnz(axis=1))
    # weighted average of cpg density
    comp = matrix.dot(cpg_ratio.values[:, None])[:, 0]

    if calc_strength:
        scores = compartment_strength(matrix, comp, cpg_ratio)
    else:
        scores = None
    return comp, scores


import scipy.sparse as sparse

def single_cell_compartment(cell_url, cell_index, cpg_profile, calc_strength, output_prefix, mode, chrom_sizes, resolution):
    if mode == 'cool':
        cool = cooler.Cooler(cell_url)
        chroms = pd.Index(cool.chromnames).intersection(cpg_profile['chrom'])
    elif mode == 'npy':
        chroms = chrom_sizes.index
    all_comp = []
    scores = np.array([0, 0, 0])
    chroms = chroms[~chroms.isin(['chrX', 'chrY'])]
    for chrom in chroms:
        cpg_ratio = cpg_profile.loc[cpg_profile['chrom'] == chrom, 'cpg_ratio']
        if mode == 'cool':
            matrix = cool.matrix(balance=False, sparse=True).fetch(chrom)
        elif mode == 'npy':
            if 'bulk' in output_prefix:
                base = os.path.dirname(cell_url)
                data = np.load(os.path.join(base, f"{chrom}_sparse_adj.npy"), allow_pickle=True)
            else:  # single-cell calling
                base = os.path.dirname(os.path.dirname(cell_url))
                data = np.load(os.path.join(base, 'raw', f"{chrom}_sparse_adj.npy"), allow_pickle=True)
            matrix = data[cell_index]

        if resolution == 500000:
            matrix = merge_hic_matrix(matrix.toarray(), original_bin_size=100000, target_bin_size=500000, merge_method='mean')
            matrix = sparse.csr_matrix(matrix)

        if cpg_ratio.shape[0] != matrix.shape[0]:
            print(f'ERROR : cpg_ratio and chrom_size have different shapes at {chrom}')
            return (0)

        comp, scores = single_chrom_compartment(matrix.tocsr(),
                                                cpg_ratio,
                                                calc_strength=calc_strength)
        if calc_strength:
            scores += scores
        all_comp.append(comp)
    all_comp = np.concatenate(all_comp)
    if calc_strength:
        np.savez(f'{output_prefix}.comp.npz', all_comp, scores)
    else:
        np.savez(f'{output_prefix}.comp.npz', all_comp)
    return


def aggregate_compartment(cell_table, temp_dir, bins, output_path, calc_strength):
    total_comp = []
    total_scores = []
    for cell_id, cell_url in cell_table.items():
        comp_path = f'{temp_dir}/{cell_id}.comp.npz'
        npz_data = np.load(comp_path, allow_pickle=True)
        total_comp.append(npz_data['arr_0'])
        if calc_strength:
            total_scores.append(npz_data['arr_1'])

    total_comp = np.vstack(total_comp)
    total_comp = pd.DataFrame(total_comp,
                              index=cell_table.index,
                              columns=bins.index)
    total_comp.index.name = 'cell'
    total_comp.columns.name = 'bin'
    total_comp = xr.Dataset({'compartment': total_comp})
    # coordinates of compartments
    total_comp.coords['bin_chrom'] = bins['chrom']
    total_comp.coords['bin_start'] = bins['start']
    total_comp.coords['bin_end'] = bins['end']

    if calc_strength:
        total_scores = np.vstack(total_scores)
        total_scores = pd.DataFrame(total_scores,
                                    index=cell_table.index,
                                    columns=['AA', 'BB', 'AB'])
        total_scores.index.name = 'cell'
        total_scores.columns.name = 'sum'
    total_comp['summary'] = xr.DataArray(total_scores)
    total_comp.to_netcdf(output_path)
    return


def multiple_cell_compartment(cell_table_path,
                              output_prefix,
                              cpg_profile_path,
                              chrom_size_path,
                              resolution,
                              cpu=10,
                              calc_strength=False,
                              mode='cool'):
    cell_table = pd.read_csv(cell_table_path,
                             sep='\t',
                             index_col=0,
                             header=None).squeeze(axis=1)
    print(cell_table.shape[0], 'cells to calculate.')

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

    # calculate individual cells
    with ProcessPoolExecutor(cpu) as exe:
        future_dict = {}
        for cell_index, (cell_id, cell_url) in enumerate(cell_table.items()):
            cell_prefix = f'{temp_dir}/{cell_id}'
            if pathlib.Path(f'{cell_prefix}.comp.npz').exists():
                continue
            future = exe.submit(single_cell_compartment,
                                cell_url=cell_url,
                                cell_index=cell_index,
                                cpg_profile=cpg_profile,
                                output_prefix=cell_prefix,
                                calc_strength=calc_strength,
                                mode=mode,
                                chrom_sizes=chrom_sizes, resolution=resolution)
            future_dict[future] = cell_id

        for future in tqdm(as_completed(future_dict), total=len(future_dict), desc="Processing cells"):
            cell_id = future_dict[future]
            try:
                future.result()
                print(f'{cell_id} finished.')
            except Exception as e:
                print(f'{cell_id} failed with error: {e}')

    aggregate_compartment(cell_table=cell_table,
                          temp_dir=temp_dir,
                          bins=bins,
                          output_path=f'{output_prefix}.compartment.nc',
                          calc_strength=calc_strength)

    # cleanup
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
        cpu=args.cpu,
        mode=args.mode,
        resolution=args.resolution,
    )

    print('done')


if __name__ == '__main__':
    main()