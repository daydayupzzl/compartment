# Compartment Calling Benchmark

**Benchmarking single-cell Hi-C A/B compartment calling**

This repository contains the code for the **compartment** part of a larger project. It systematically benchmarks four mainstream single-cell Hi-C A/B compartment calling methods across raw data and two imputation strategies (scHiCluster / Higashi), at both single-cell and pseudo-bulk levels, and at 100 kb / 500 kb resolutions.

---

## 0. Project Background & Data Source

The larger project is split into four relatively independent parts, each handled by a different collaborator:

| Part | Description |
| --- | --- |
| impute | Imputation of single-cell Hi-C contact matrices |
| loop | Loop calling |
| tad | TAD / domain calling |
| **compartment** | A/B compartment calling, i.e. this repository |

**All input data for this repository comes from the outputs of the impute part:**

- **raw**: un-imputed raw single-cell Hi-C contact matrices;
- **scHiCluster imputed**: contact matrices imputed by scHiCluster;
- **Higashi imputed**: contact matrices imputed by Higashi.

> Before running this repository, please refer to the repository of the **impute** part to complete data preprocessing and imputation and obtain the three input modalities above. This repository is responsible only for compartment calling and evaluation; it does not include data preprocessing or imputation code.

---

## 1. Repository Structure

```
paper-code
├── README.md
├── CpG/              # CpG method
├── Self-PCA/         # Self-PCA method
├── Ref-PCA/          # Ref-PCA method (includes Higashi-related modules)
└── Evaluation/       # Evaluation code
```

---

## 2. Environment Setup

Most of the code in this repository builds on the environment and source code of **Higashi** and **scHiCluster**, so we recommend setting up environments per method.

### 2.1 scHiCluster Environment

Used for CpG, Self-PCA, and most of the evaluation. Relies on `numpy / scipy / pandas / h5py / cooler / scikit-learn / joblib / xarray / tables`, and the CpG calculation additionally requires `bedtools`.

```bash
conda create -n schicluster python==3.6.8
conda activate schicluster
pip install git+https://github.com/zhoujt1994/scHiCluster.git
```

The CpG method is adapted from the built-in compartment module of scHiCluster, following the same idea, split out and rewritten to fit our evaluation pipeline.

### 2.2 Higashi Environment

**Ref-PCA is directly derived from the Higashi source code** and depends on Higashi's built-in analysis and backend modules. The simplest way is to put the whole `Ref-PCA` directory into the Higashi source tree and run it in the Higashi environment.

```bash
conda install -c ruochiz fasthigashi   # or
git clone https://github.com/ma-compbio/Higashi/
cd Higashi
python setup.py install
```

Reference: https://github.com/ma-compbio/Higashi

### 2.3 scGHOST Environment

Used for the optional comparison at 500 kb. **This repository does not provide the scGHOST source code**; please refer to its original repository for installation and usage:

Reference: https://github.com/ma-compbio/scGHOST

---

## 3. Data Description

### 3.1 Input Data

Each dataset provides three modalities: raw, scHiCluster imputed, and Higashi imputed. Matrices can be read either from `.cool` format or from NumPy sparse graphs.

### 3.2 Data Scenarios & Resolutions

- **Single-cell**: per-cell A/B calling;
- **Pseudo-bulk**: single-cell matrices of the same cell type are merged in varying numbers, e.g. bulk100, bulk200, …, bulkall.

- **100 kb**: compares CpG, Self-PCA, Ref-PCA;
- **500 kb**: produced by bin-merging the 100 kb matrices, comparing all four methods (including scGHOST). scGHOST is extremely compute- and memory-intensive, making it impractical to run at 100 kb on typical hardware, so it is only included at 500 kb.

### 3.3 Ground Truth Labels

The ground truth uses the combination of **raw + bulkall + PCA**: all raw single cells of the cell type are merged into a bulk matrix, and A/B is called with classic PCA as an unbiased reference. For boundary-related metrics, a sign change in A/B is treated as a boundary.

---

## 4. Methods

The four methods span three major paradigms: linear dimensionality reduction, epigenetic prior, and graph embedding.

### 4.1 CpG — Epigenetic Prior

Adapted from the CpG compartment code of scHiCluster. It exploits the coupling between intrinsic genomic CpG density and chromatin openness, combining per-bin CpG density with Hi-C signal through a linear operation to directly obtain A/B values per bin. The method is lightweight, requires no matrix decomposition, and is robust on sparse single cells.

### 4.2 Self-PCA — Linear Dimensionality Reduction

Performs a standard PCA compartment calling on each target matrix independently, serving as the field baseline: visibility normalization and distance-decay correction are applied first, followed by a bin-to-bin correlation matrix and PCA; the sign is then calibrated by CpG correlation, and a better principal component is selected by compartment structure strength as the final A/B value.

### 4.3 Ref-PCA — Reference-Constrained PCA

Adapted from the Higashi source code, Ref-PCA introduces a **reference model** on top of Self-PCA: a set of PCA models is first trained on an overall merged bulk, and each single cell or pseudo-bulk sample is then projected onto these pre-trained feature axes to mitigate dimensionality drift caused by low-quality samples. Chromosomes are also split into p / q arms at the centromere before calling.

### 4.4 scGHOST — Graph Embedding

A deep graph-embedding method that models genomic bins as graph vertices and interaction strengths as edge weights, using random walks and clustering to obtain sub-compartments such as A1/A2/B1/B2/B3, which are then mapped to final A/B values.

- **Single-cell**: called per cell following the original repository;
- **Pseudo-bulk**: scGHOST is designed for single cells and has no native bulk module. Here we adopt post-hoc aggregation: call each single cell first, then bin-level average within each bulk group to obtain bulk-level compartment results.

---

## 5. Usage

> **Important**: this repository provides **core calling code only** and does not include dataset-specific glue or logging code; input and output paths are written relative to a `data/` directory (Hi-C matrices, compartment calls, labels) and a `files/` directory (chromosome sizes, CpG profiles). Please adapt them to your own data layout before running.

Each method exposes a command-line entry point; for calling conventions and input arguments, see the in-code comments in each directory. The overall workflow is roughly:

```bash
# CpG / Self-PCA
python CpG/calling.py --dataset <dataset> --resolution 100000 ...
python Self-PCA/calling.py --dataset <dataset> --resolution 100000 ...

# Ref-PCA (run inside the Higashi environment and source tree)
python Ref-PCA/calling_single_cell.py -c <config.json> ...
python Ref-PCA/calling_bulk.py      -c <config.json> ...

# scGHOST (refer to the original repository)
python scghost.py --config <config.json>

# Evaluation (per dataset x cell type)
python Evaluation/get_data.py <dataset> <cell_type> --resolution 100000   # or 500000 (adds scGHOST)
#   then run Evaluation/get_result.ipynb; its last cell exports metrics to Evaluation/metrics_out/
python Evaluation/clustering.py --compartment <calls.hdf5|.nc> --labels label_info.pickle \
       --dataset <dataset> --method <PCA|CPG|Ref-PCA> --data_type <Raw|scHiCluster|Higashi> \
       --out Evaluation/cluster/100k.csv                   # one call per method x data type

# Ranking (see Section 6.2)
python Evaluation/collect_metrics.py --setting sc_100k --cluster Evaluation/cluster/100k.csv
python Evaluation/collect_metrics.py --setting bulk300_100k
python Evaluation/ranking.py
```

The input to each method is the three Hi-C matrix modalities described above, and the output is a per-cell A/B compartment vector file (CpG / Self-PCA output in an established readable format; Ref-PCA outputs hdf5). The evaluation entry point reads these results together with the ground-truth labels and produces the metrics described in the next section.

---

## 6. Evaluation Metrics

Evaluation is performed independently at 100 kb / 500 kb resolutions and across the raw / scHiCluster / Higashi modalities:

- **Boundary accuracy**: overlap between the A/B boundaries detected by each algorithm and the ground-truth boundaries, with 1-bin tolerance, reported as Precision / Recall / F1 / Jaccard, along with the number of boundaries.
- **Label correlation**: Pearson correlation between the compartment vector output by an algorithm and the ground-truth label vector.
- **Cell clustering**: the per-cell genome-wide A/B values are composed into a cell feature matrix and clustered, using ARI, AMI, V-Measure, and Calinski-Harabasz to assess how well the compartment features distinguish cell types.
- **Compartment strength**: saddle strength of A/B computed on the corresponding Hi-C matrix, quantifying the spatial separation between active and inactive chromatin.

The evaluation also provides a denoising step to filter out fragmented tiny segments caused by noise, making boundary detection more robust.

### 6.1 Aggregation of per-metric values

- **Cell type**: each dataset is evaluated on **one representative cell type**, fixed for all analyses of that dataset (`<cell_type>` argument of `get_data.py`). The ground truth (bulk-all) and all pseudo-bulk samples are built from this cell type. The clustering metrics are the exception: they use all cells of the dataset with all annotated cell types as reference labels.

  | Dataset | Representative cell type |
  | --- | --- |
  | Tan | Neuron |
  | Lee | ODC |
  | Tian | SubCtx-Cplx |
  | Liu | mix late mesenchyme |
  | Zhou | L2/3 |
  | Chang2024_droplot | ITL23GL |
  | Chang2024_paired | ITL23GL |

- **Single-cell**: for each cell, metrics are averaged over chromosomes; the per-cell values are then averaged over all cells of the representative cell type (`compute_single_cell_overall_metrics` in `get_result.ipynb`).
- **Pseudo-bulk**: metrics are averaged over chromosomes of the pseudo-bulk sample (`compute_bulk_overall_metrics`). The ranking figures use **bulk300**.
- **Clustering** (`Evaluation/clustering.py`): the genome-wide compartment vectors of all cells form a cell × bin matrix, which is reduced by PCA (512 components, `random_state=0`) and clustered by KMeans with *k* = number of annotated cell types (`random_state=0`, `n_init=10`; set explicitly because scikit-learn ≥ 1.4 changed the default). ARI / AMI / V-Measure are computed against the annotation and CHI on the PCA embedding.

### 6.2 Comprehensive ranking (`Evaluation/ranking.py`)

The method rankings in the paper (Fig. 1a/1c, Fig. 3c/d) are **rank-based**:

1. **Comparison group**: one (scenario, resolution) pair — single-cell @ 100 kb, single-cell @ 500 kb, bulk300 @ 100 kb, bulk300 @ 500 kb. Groups are ranked independently. Within a group, all *method × data type* combinations are compared (9 at 100 kb; 12 at 500 kb including scGHOST).
2. **Normalisation**: for each metric and each dataset, combinations are ranked in descending order (larger is better, ties take the minimum rank). Ranks are dimensionless, so no further scaling of F1 / Pearson / Strength / ARI is applied.
3. **Aggregation**: per metric, ranks are averaged over the 7 datasets (rank first, then aggregate).
4. **Composite score**: `S = 0.4·R̄(F1) + 0.3·R̄(Correlation) + 0.3·R̄(Strength)` for pseudo-bulk and `S = 0.4·R̄(F1) + 0.3·R̄(Correlation) + 0.3·R̄(ARI)` for single-cell. Lower is better; the overall rank is the order of `S`. Boundary count is displayed but has weight 0.

The input is `Evaluation/ranking_data/<setting>.csv`, one row per dataset × method × data type, with columns `dataset, method, data_type, F1, Correlation, Count` and `ARI` (`sc_*`) or `Strength` (`bulk*`). It is assembled by `collect_metrics.py` from the metrics exported by `get_result.ipynb` and the clustering results of `clustering.py` (see Section 5).

```bash
python Evaluation/ranking.py   # writes Evaluation/ranking_out/<setting>_per_dataset.csv and <setting>_ranking.csv
```

---

## 7. FAQ

1. Paths in the code are relative to `data/` and `files/`; please adapt them to your own data layout. For Ref-PCA, an optional `raw_dir` entry in the config overrides the directory of the raw matrices (default: `temp_dir`).
2. Ref-PCA depends on the Higashi source code; please run it in the Higashi environment.
3. scGHOST is compared only at 500 kb because its memory usage at 100 kb is extremely high and impractical on typical hardware.
4. CpG profiles and chromosome-size files are preconfigured for hg38 / mm10; please update them for your reference genome.

---

## 8. References

- Higashi: https://github.com/ma-compbio/Higashi
- scHiCluster: https://github.com/zhoujt1994/scHiCluster
- scGHOST: https://github.com/ma-compbio/scGHOST
- Input data (raw / two imputed modalities): please refer to the repository of the **impute** part of the larger project.