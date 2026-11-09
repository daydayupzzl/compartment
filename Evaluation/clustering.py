import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, calinski_harabasz_score

OUT_COLS = ['method', 'dataset', 'data_type', 'ARI', 'AMI', 'V-measure', 'CHI']


def clustering_scores(comp, cell_types, n_pca=512, seed=0):
    comp = np.asarray(comp, dtype=np.float64)
    cell_types = np.asarray(cell_types)
    if comp.ndim != 2 or comp.shape[0] != len(cell_types):
        raise ValueError(f'compartment matrix {comp.shape} does not match {len(cell_types)} labels')
    if np.isnan(comp).any():
        raise ValueError(f'compartment matrix contains {np.isnan(comp).sum()} NaN values')

    emb = PCA(n_components=min(n_pca, *comp.shape), random_state=seed).fit_transform(comp)
    # n_init fixed to 10 (scikit-learn >= 1.4 changed the default to 'auto')
    pred = KMeans(n_clusters=len(np.unique(cell_types)), random_state=seed, n_init=10).fit(emb).labels_
    chi = calinski_harabasz_score(emb, pred) if len(set(pred)) > 1 else np.nan
    return {'ARI': adjusted_rand_score(cell_types, pred),
            'AMI': adjusted_mutual_info_score(cell_types, pred),
            'V-measure': v_measure_score(cell_types, pred),
            'CHI': chi}


def load_compartment_matrix(path, n_cells):
    if path.endswith(('.hdf5', '.h5')):
        import h5py
        with h5py.File(path, 'r') as f:
            cp = f['compartment']
            missing = [i for i in range(n_cells) if f'cell_{i}' not in cp]
            if missing:
                raise KeyError(f'{path}: {len(missing)} cells missing (first: cell_{missing[0]})')
            return np.stack([np.asarray(cp[f'cell_{i}']) for i in range(n_cells)])
    if path.endswith('.nc'):
        import xarray as xr
        with xr.open_dataset(path) as ds:
            mat = ds['compartment'].transpose('cell', 'bin').values
        if mat.shape[0] != n_cells:
            raise ValueError(f'{path}: {mat.shape[0]} cells but {n_cells} labels')
        return mat
    raise ValueError(f'unsupported compartment file: {path}')


def upsert_row(out_csv, row):
    df = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=OUT_COLS)
    for c in ['method', 'dataset', 'data_type']:
        df[c] = df[c].astype(str).str.strip()
    key = (df.method == row['method']) & (df.dataset == row['dataset']) & (df.data_type == row['data_type'])
    df = pd.concat([df[~key], pd.DataFrame([row])], ignore_index=True)[OUT_COLS]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description='Cell-type clustering metrics from single-cell compartment calls')
    parser.add_argument('--compartment', required=True, help='compartment file (.hdf5 for Ref-PCA, .nc for PCA/CpG)')
    parser.add_argument('--labels', required=True, help='label_info.pickle with a "cell type" key')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--method', required=True, help='PCA / CPG / Ref-PCA / scGHOST')
    parser.add_argument('--data_type', required=True, help='Raw / scHiCluster / Higashi')
    parser.add_argument('--out', required=True, help='output csv, e.g. cluster/100k.csv')
    args = parser.parse_args(argv)

    with open(args.labels, 'rb') as f:
        cell_types = np.asarray(pickle.load(f)['cell type'])
    comp = load_compartment_matrix(args.compartment, len(cell_types))
    scores = clustering_scores(comp, cell_types)
    row = {'method': args.method, 'dataset': args.dataset, 'data_type': args.data_type,
           **{k: round(float(v), 4) for k, v in scores.items()}}
    upsert_row(args.out, row)
    print(row)


if __name__ == '__main__':
    main()
