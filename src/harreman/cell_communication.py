import ast
import itertools
import json
import multiprocessing
from functools import partial
from random import random, sample
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hcluster
import sparse
import tensorly as tl
import torch
from anndata import AnnData
from numba import jit, njit
from scanpy.preprocessing._utils import _get_mean_var
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from . import models
from .anndata import setup_anndata
from .database import counts_from_anndata
from .knn import compute_neighbors, compute_neighbors_from_distances, compute_node_degree_ct_pair, compute_weights
from .local_autocorrelation import compute_local_autocorrelation_fast, compute_local_cov_max


def compute_gene_pairs(
    adata, database_varm_key, layer_key, cell_type_key, autocorrelation_filt, expression_filt, de_filt, cell_type_pairs=None
):
    from_value_to_type = {
        -1.0: "IMP/REC",
        1.0: "EXP/LIG",
        2.0: "IMP/EXP",
    }

    use_raw = layer_key == "use_raw"
    genes = adata.raw.var.index if use_raw else adata.var_names

    cell_types = adata.obs[cell_type_key] if not use_raw else adata.raw.obs[cell_type_key]
    cell_types = cell_types.values.astype(str)

    database = adata.varm[database_varm_key]

    if (expression_filt is True) or (de_filt is True):
        filtered_genes = adata.uns["filtered_genes"]
        filtered_genes_ct = adata.uns["filtered_genes_ct"]
    elif autocorrelation_filt is True:
        autocor_results = adata.uns["gene_autocorrelation_results"]
        filtered_genes = autocor_results[autocor_results.FDR < 0.05].index.tolist()
    else:
        filtered_genes = genes

    if "filtered_genes_ct" not in adata.uns:
        filtered_genes_ct = {}
        for ct in cell_types:
            filtered_genes_ct[ct] = filtered_genes

    if len(filtered_genes) == 0:
        raise ValueError("No genes have passed the filters.")

    non_sig_genes = [g for g in genes if g not in filtered_genes]

    database.loc[non_sig_genes] = 0
    adata.varm[database_varm_key] = database

    gene_pairs_per_metabolite = {}
    gene_pairs = []
    ct_pairs = []

    weights = adata.obsp["weights"]

    if cell_type_pairs is None:
        cell_type_list = list(filtered_genes_ct.keys())
        cell_type_pairs = list(itertools.combinations(cell_type_list, 2))

    cell_type_pairs_df = pd.Series(cell_type_pairs)
    cell_type_pairs_int = cell_type_pairs_df.apply(get_interacting_cell_type_pairs, args=(weights, cell_types))
    cell_type_pairs = cell_type_pairs_df[cell_type_pairs_int].tolist()
    cell_type_pairs = sample(cell_type_pairs, 5)
    gene_pairs_per_ct_pair = {}

    for metabolite in database.columns:
        metab_genes = database[database[metabolite] != 0].index.tolist()
        if len(metab_genes) == 0:
            continue
        gene_pairs_per_metabolite[metabolite] = {"gene_pair": [], "gene_type": []}
        # for each metabolite, we compute all the possible gene pairs. Then, we remove those pairs where both genes are either importers/receptor or exporters/ligands.
        all_pairs = list(set(itertools.combinations_with_replacement(metab_genes, 2)) | set(itertools.permutations(metab_genes, 2)))
        for pair in all_pairs:
            var1, var2 = pair
            var1_value = database.loc[var1, metabolite]
            var2_value = database.loc[var2, metabolite]
            if not (var1_value == 1.0 and var2_value == 1.0) or (var1_value == -1.0 and var2_value == -1.0):
                var1_type = from_value_to_type[var1_value]
                var2_type = from_value_to_type[var2_value]
                gene_pairs_per_metabolite[metabolite]["gene_pair"].append((var1, var2))
                gene_pairs_per_metabolite[metabolite]["gene_type"].append((var1_type, var2_type))
                if (var1, var2) not in gene_pairs: # ((var1, var2) not in gene_pairs) and ((var2, var1) not in gene_pairs)
                    gene_pairs.append(pair)
                    for ct_pair in cell_type_pairs:
                        ct_1, ct_2 = ct_pair
                        ct_pair_str = (ct_1, ct_2)
                        if (var1 in filtered_genes_ct[ct_1]) and (var2 in filtered_genes_ct[ct_2]):
                            if ct_pair_str not in gene_pairs_per_ct_pair.keys():
                                gene_pairs_per_ct_pair[ct_pair_str] = []
                            gene_pairs_per_ct_pair[ct_pair_str].append(pair)
                            if ct_pair not in ct_pairs:
                                ct_pairs.append(ct_pair)

    if "gene_pairs" not in adata.uns:
        adata.uns["gene_pairs"] = gene_pairs
    if "cell_type_pairs" not in adata.uns:
        adata.uns["cell_type_pairs"] = ct_pairs
    if "gene_pairs_per_metabolite" not in adata.uns:
        adata.uns["gene_pairs_per_metabolite"] = gene_pairs_per_metabolite
    if "gene_pairs_per_ct_pair" not in adata.uns:
        adata.uns["gene_pairs_per_ct_pair"] = gene_pairs_per_ct_pair


def compute_cell_communication_p(
    adata,
    layer_key,
    database_varm_key,
    model,
    cell_type_key,
    cell_type_list,
    jobs,
):
    use_raw = layer_key == "use_raw"
    database = adata.varm[database_varm_key] if not use_raw else adata.raw.varm[database_varm_key]

    cells = adata.obs_names if not use_raw else adata.raw.obs.index
    cells = cells.values.astype(str)

    # We select genes that are associated with at least one metabolite
    genes = database.loc[(database != 0).any(axis=1)].index

    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)

    weights = adata.obsp["weights"]
    cell_types = adata.obs[cell_type_key] if not use_raw else adata.raw.obs[cell_type_key]
    num_umi = counts.sum(axis=0)
    gene_pairs = adata.uns["gene_pairs"]
    del adata.uns["gene_pairs"]
    cell_type_pairs = adata.uns["cell_type_pairs"]
    gene_pairs_per_ct_pair = adata.uns["gene_pairs_per_ct_pair"]
    del adata.uns["gene_pairs_per_ct_pair"]

    gene_pairs_ind = []
    for pair in gene_pairs:
        var1, var2 = pair
        pair_tuple = (genes.to_list().index(var1), genes.to_list().index(var2))
        gene_pairs_ind.append(pair_tuple)

    # adata.uns['gene_pairs_ind'] = gene_pairs_ind

    gene_pairs_per_ct_pair_ind = {}
    for ct_pair in gene_pairs_per_ct_pair.keys():
        gene_pairs = gene_pairs_per_ct_pair[ct_pair]
        gene_pairs_per_ct_pair_ind[ct_pair] = []
        for pair in gene_pairs:
            var1, var2 = pair
            pair_tuple = (genes.to_list().index(var1), genes.to_list().index(var2))
            gene_pairs_per_ct_pair_ind[ct_pair].append(pair_tuple)

    # adata.uns['gene_pairs_per_ct_pair_ind'] = gene_pairs_per_ct_pair_ind

    # Compute node degree
    print(f"Computing node degree for {len(cell_type_pairs)} cell type pairs...")

    # D = []
    # for ct_pair_i in tqdm(range(len(cell_type_pairs))):
    #     ct_pair = cell_type_pairs[ct_pair_i]
    #     ct_t, ct_u = ct_pair
    #     cell_type_mask_t = cell_types == ct_t
    #     cell_type_mask_t = cell_type_mask_t.reset_index(drop=True)
    #     cell_type_mask_t_ind = cell_type_mask_t[cell_type_mask_t].index.tolist()
    #     cell_type_mask_u = cell_types == ct_u
    #     cell_type_mask_u = cell_type_mask_u.reset_index(drop=True)
    #     cell_type_mask_u_ind = cell_type_mask_u[cell_type_mask_u].index.tolist()
    #     weights_ct = weights[cell_type_mask_t,]

    #     D_ct_pair = compute_node_degree_ct_pair(weights_ct.toarray(), cell_type_mask_t_ind, cell_type_mask_u_ind)
    #     D.append(D_ct_pair)

    counts = create_centered_counts(counts, model, num_umi)
    counts = np.nan_to_num(counts)

    # Compute the expectation of H squared
    print(f"Computing the expectation of H squared for {len(cell_type_pairs)} cell type pairs...")

    eg2s = np.array(
        [
            conditional_eg2_cellcom(cell_type_pairs[i], counts, weights, cell_types.tolist())
            for i in tqdm(range(len(cell_type_pairs)))
        ]
    )

    def initializer():
        # global g_neighbors
        global g_weights
        global g_counts
        global g_eg2s
        global g_cell_types
        global g_cell_type_list
        global g_cell_type_pairs
        global g_gene_pairs_ind
        global g_gene_pairs_per_ct_pair_ind
        global g_genes
        global g_cells
        g_counts = counts
        # g_neighbors = neighbors
        g_weights = weights
        g_eg2s = eg2s
        g_cell_types = cell_types
        g_cell_type_list = cell_type_list
        g_cell_type_pairs = cell_type_pairs
        g_gene_pairs_ind = gene_pairs_ind
        g_gene_pairs_per_ct_pair_ind = gene_pairs_per_ct_pair_ind
        g_genes = genes
        g_cells = cells

    print(f"Computing cell-cell communication for {len(cell_type_pairs)} cell type pairs...")

    if jobs > 1:
        with multiprocessing.Pool(processes=jobs, initializer=initializer) as pool:
            results = list(
                tqdm(
                    pool.imap(_map_fun_parallel_pairs_centered_cond_cellcom_p, range(len(cell_type_pairs))),
                    total=len(cell_type_pairs),
                )
            )
    else:

        def _map_fun(ct_pair_i):
            return _compute_hs_pairs_inner_centered_cond_sym_cellcom_p(
                ct_pair_i,
                counts,
                weights,
                eg2s,
                cell_types,
                cell_type_pairs,
                gene_pairs_ind,
                gene_pairs_per_ct_pair_ind,
            )

        results = list(tqdm(map(_map_fun, range(len(cell_type_pairs))), total=len(cell_type_pairs)))

    cell_communication_results = get_cell_communication_results_p(
        counts,
        results,
        gene_pairs_per_ct_pair,
        gene_pairs_per_ct_pair_ind,
        cell_types.tolist(),
        cell_type_pairs,
        gene_pairs_ind,
    )

    cell_communication_df = cell_communication_results[0]

    adata.uns["cell_communication_df"] = cell_communication_df


def compute_cell_communication_p_fast(
    adata,
    layer_key,
    database_varm_key,
    model,
    cell_type_key,
    cell_type_list,
):
    print("Starting the cell-cell communication analysis...")

    use_raw = layer_key == "use_raw"
    database = adata.varm[database_varm_key] if not use_raw else adata.raw.varm[database_varm_key]

    cells = adata.obs_names if not use_raw else adata.raw.obs.index
    cells = cells.values.astype(str)

    # We select genes that are associated with at least one metabolite
    genes = database.loc[(database != 0).any(axis=1)].index

    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)

    weights = adata.obsp["weights"]
    cell_types = adata.obs[cell_type_key] if not use_raw else adata.raw.obs[cell_type_key]
    num_umi = counts.sum(axis=0)
    gene_pairs = adata.uns["gene_pairs"]
    # del adata.uns["gene_pairs"]
    cell_type_pairs = adata.uns["cell_type_pairs"]
    gene_pairs_per_ct_pair = adata.uns["gene_pairs_per_ct_pair"]
    # del adata.uns["gene_pairs_per_ct_pair"]

    gene_pairs_ind = []
    for pair in gene_pairs:
        var1, var2 = pair
        pair_tuple = (genes.to_list().index(var1), genes.to_list().index(var2))
        gene_pairs_ind.append(pair_tuple)

    # adata.uns['gene_pairs_ind'] = gene_pairs_ind

    gene_pairs_per_ct_pair_ind = {}
    for ct_pair in gene_pairs_per_ct_pair.keys():
        gene_pairs = gene_pairs_per_ct_pair[ct_pair]
        gene_pairs_per_ct_pair_ind[ct_pair] = []
        for pair in gene_pairs:
            var1, var2 = pair
            pair_tuple = (genes.to_list().index(var1), genes.to_list().index(var2))
            gene_pairs_per_ct_pair_ind[ct_pair].append(pair_tuple)

    # adata.uns['gene_pairs_per_ct_pair_ind'] = gene_pairs_per_ct_pair_ind

    # Compute node degree

    D = compute_node_degree_ct_pair(weights, cell_type_pairs, cell_types)

    # print(f"Computing node degree for {len(cell_type_pairs)} cell type pairs...")

    # D = []
    # for ct_pair_i in tqdm(range(len(cell_type_pairs))):
    #     ct_pair = cell_type_pairs[ct_pair_i]
    #     ct_t, ct_u = ct_pair
    #     cell_type_mask_t = cell_types == ct_t
    #     cell_type_mask_t_df = cell_type_mask_t.reset_index(drop=True)
    #     cell_type_mask_t_ind = cell_type_mask_t_df[cell_type_mask_t_df].index.tolist()
    #     cell_type_mask_u = cell_types == ct_u
    #     cell_type_mask_u_df = cell_type_mask_u.reset_index(drop=True)
    #     cell_type_mask_u_ind = cell_type_mask_u_df[cell_type_mask_u_df].index.tolist()
    #     weights_ct = weights[cell_type_mask_t,]

    #     D_ct_pair = compute_node_degree_ct_pair(weights_ct.toarray(), cell_type_mask_t_ind, cell_type_mask_u_ind)
    #     D.append(D_ct_pair)

    counts = create_centered_counts(counts, model, num_umi)
    counts = np.nan_to_num(counts)
    counts = sparse.COO.from_numpy(counts)

    # Generate the 3D tensor (both counts and weights) adapting for each cell type pair
    # The idea would be to first replicate the 2D array N times into a 3rd dimension (or 4 in case of having multiple measurements)
    # And then modify each 2D matrix independently. For this, we need the list of cell types for each cell, the list of cell type pairs, and the genes per ct pair ind dictionary

    print("Getting the cell-type-pair-specific counts and weights...")

    counts_ct_pairs_t, counts_ct_pairs_u, weigths_ct_pairs = get_ct_pair_counts_and_weights(
        counts, weights, cell_type_pairs, cell_types, gene_pairs_per_ct_pair_ind
    )

    print(f"Computing the expectation of H squared for {len(cell_type_pairs)} cell type pairs...")

    eg2s = conditional_eg2_cellcom_fast(counts_ct_pairs_t, counts_ct_pairs_u, weigths_ct_pairs)

    print(f"Computing cell-cell communication for {len(cell_type_pairs)} cell type pairs...")

    results = compute_cellcom_p(
        counts_ct_pairs_t,
        counts_ct_pairs_u,
        weigths_ct_pairs,
        eg2s,
        cell_types,
        cell_type_pairs,
        gene_pairs_ind,
        gene_pairs_per_ct_pair_ind,
    )

    print("Obtaining the communication results...")

    cellcom_results = get_cell_communication_results(
        counts,
        results,
        gene_pairs_per_ct_pair,
        gene_pairs_per_ct_pair_ind,
        cell_types.tolist(),
        cell_type_pairs,
        gene_pairs_ind,
        D,
        test='parametric',
    )

    adata.uns["cell_communication_df"] = cellcom_results[0]
    adata.uns["lcs_3d"] = cellcom_results[1]
    adata.uns["lc_zs_3d"] = cellcom_results[2]
    adata.uns["gene_pairs_ind_new"] = cellcom_results[3]
    adata.uns["cell_type_list"] = cellcom_results[4]
    adata.uns["genes"] = genes


def compute_cell_communication_np(
    adata,
    layer_key,
    database_varm_key,
    cell_type_key,
    jobs,
    M=1000,
):
    use_raw = layer_key == "use_raw"
    database = adata.varm[database_varm_key] if not use_raw else adata.raw.varm[database_varm_key]

    cells = adata.obs_names if not use_raw else adata.raw.obs.index
    cells = cells.values.astype(str)

    # We select genes that are associated with at least one metabolite
    genes = database.loc[(database != 0).any(axis=1)].index

    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)

    weights = adata.obsp["weights"]
    cell_types = adata.obs[cell_type_key] if not use_raw else adata.raw.obs[cell_type_key]
    gene_pairs = adata.uns["gene_pairs"]
    del adata.uns["gene_pairs"]
    cell_type_pairs = adata.uns["cell_type_pairs"]
    gene_pairs_per_ct_pair = adata.uns["gene_pairs_per_ct_pair"]
    del adata.uns["gene_pairs_per_ct_pair"]

    gene_pairs_per_ct_pair_ind = {}
    for ct_pair in gene_pairs_per_ct_pair.keys():
        gene_pairs = gene_pairs_per_ct_pair[ct_pair]
        gene_pairs_per_ct_pair_ind[ct_pair] = []
        for pair in gene_pairs:
            var1, var2 = pair
            pair_tuple = (genes.to_list().index(var1), genes.to_list().index(var2))
            gene_pairs_per_ct_pair_ind[ct_pair].append(pair_tuple)

    # adata.uns['gene_pairs_per_ct_pair_ind'] = gene_pairs_per_ct_pair_ind

    # Compute node degree
    print(f"Computing node degree for {len(cell_type_pairs)} cell type pairs...")

    D = []
    for ct_pair_i in tqdm(range(len(cell_type_pairs))):
        ct_pair = cell_type_pairs[ct_pair_i]
        ct_t, ct_u = ct_pair
        cell_type_mask_t = cell_types == ct_t
        cell_type_mask_t = cell_type_mask_t.reset_index(drop=True)
        cell_type_mask_t_ind = cell_type_mask_t[cell_type_mask_t].index.tolist()
        cell_type_mask_u = cell_types == ct_u
        cell_type_mask_u = cell_type_mask_u.reset_index(drop=True)
        cell_type_mask_u_ind = cell_type_mask_u[cell_type_mask_u].index.tolist()
        weights_ct = weights[cell_type_mask_t,]

        D_ct_pair = compute_node_degree_ct_pair(weights_ct.toarray(), cell_type_mask_t_ind, cell_type_mask_u_ind)
        D.append(D_ct_pair)

    def initializer():
        global g_weights
        global g_counts
        global g_cell_types
        global g_gene_pairs_per_ct_pair_ind
        global g_genes
        global g_cells
        global g_M
        g_weights = weights
        g_counts = counts
        g_cell_types = cell_types
        g_gene_pairs_per_ct_pair_ind = gene_pairs_per_ct_pair_ind
        g_genes = genes
        g_cells = cells
        g_M = M

    print(f"Computing cell-cell communication for {len(cell_type_pairs)} cell type pairs...")

    if jobs > 1:
        with multiprocessing.Pool(processes=jobs, initializer=initializer) as pool:
            results = list(
                tqdm(
                    pool.imap(_map_fun_parallel_pairs_centered_cond_cellcom_np, cell_type_pairs),
                    total=len(cell_type_pairs),
                )
            )
    else:

        def _map_fun(ct_pair):
            return _compute_hs_pairs_inner_centered_cond_sym_cellcom_np(
                ct_pair, weights, counts, cell_types, gene_pairs_per_ct_pair_ind, genes, cells, M
            )

        results = list(tqdm(map(_map_fun, cell_type_pairs), total=len(cell_type_pairs)))

    cell_communication_df = get_cell_communication_results_np(results, gene_pairs_per_ct_pair)

    adata.uns["cell_communication_df"] = cell_communication_df


def compute_cell_communication_np_fast(
    adata,
    layer_key,
    database_varm_key,
    cell_type_key,
    M=1000,
):
    print("Starting the cell-cell communication analysis...")

    use_raw = layer_key == "use_raw"
    database = adata.varm[database_varm_key] if not use_raw else adata.raw.varm[database_varm_key]

    cells = adata.obs_names if not use_raw else adata.raw.obs.index
    cells = cells.values.astype(str)

    # We select genes that are associated with at least one metabolite
    genes = database.loc[(database != 0).any(axis=1)].index

    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)
    counts = sparse.COO.from_numpy(counts)

    weights = adata.obsp["weights"]
    cell_types = adata.obs[cell_type_key] if not use_raw else adata.raw.obs[cell_type_key]
    gene_pairs = adata.uns["gene_pairs"]
    # del adata.uns["gene_pairs"]
    cell_type_pairs = adata.uns["cell_type_pairs"]
    gene_pairs_per_ct_pair = adata.uns["gene_pairs_per_ct_pair"]
    # del adata.uns["gene_pairs_per_ct_pair"]

    gene_pairs_ind = []
    for pair in gene_pairs:
        var1, var2 = pair
        pair_tuple = (genes.to_list().index(var1), genes.to_list().index(var2))
        gene_pairs_ind.append(pair_tuple)

    # adata.uns['gene_pairs_ind'] = gene_pairs_ind

    gene_pairs_per_ct_pair_ind = {}
    for ct_pair in gene_pairs_per_ct_pair.keys():
        gene_pairs = gene_pairs_per_ct_pair[ct_pair]
        gene_pairs_per_ct_pair_ind[ct_pair] = []
        for pair in gene_pairs:
            var1, var2 = pair
            pair_tuple = (genes.to_list().index(var1), genes.to_list().index(var2))
            gene_pairs_per_ct_pair_ind[ct_pair].append(pair_tuple)

    # adata.uns['gene_pairs_per_ct_pair_ind'] = gene_pairs_per_ct_pair_ind

    # Compute node degree

    D = compute_node_degree_ct_pair(weights, cell_type_pairs, cell_types)

    # print("Computing node degree for {} cell type pairs...".format(len(cell_type_pairs)))

    # D = []
    # for ct_pair_i in tqdm(range(len(cell_type_pairs))):
    #     ct_pair = cell_type_pairs[ct_pair_i]
    #     ct_t, ct_u = ct_pair
    #     cell_type_mask_t = cell_types == ct_t
    #     cell_type_mask_t = cell_type_mask_t.reset_index(drop=True)
    #     cell_type_mask_t_ind = cell_type_mask_t[cell_type_mask_t].index.tolist()
    #     cell_type_mask_u = cell_types == ct_u
    #     cell_type_mask_u = cell_type_mask_u.reset_index(drop=True)
    #     cell_type_mask_u_ind = cell_type_mask_u[cell_type_mask_u].index.tolist()
    #     weights_ct = weights[cell_type_mask_t,]

    #     D_ct_pair = compute_node_degree_ct_pair(weights_ct.toarray(), cell_type_mask_t_ind, cell_type_mask_u_ind)
    #     D.append(D_ct_pair)

    print("Creating the cell-type-pair-specific counts and weights...")

    counts_ct_pairs_t, counts_ct_pairs_u, weigths_ct_pairs = get_ct_pair_counts_and_weights(
        counts, weights, cell_type_pairs, cell_types, gene_pairs_per_ct_pair_ind
    )

    print("Building the null distribution...")

    counts_ct_pairs_t_null, counts_ct_pairs_u_null, weigths_ct_pairs_null = get_ct_pair_counts_and_weights_null(
        counts_ct_pairs_t, counts_ct_pairs_u, weights, cell_type_pairs, cell_types, M
    )

    print("Computing cell-cell communication for {} cell type pairs...".format(len(cell_type_pairs)))

    results = compute_cellcom_np(
        counts_ct_pairs_t,
        counts_ct_pairs_u,
        weigths_ct_pairs,
        counts_ct_pairs_t_null,
        counts_ct_pairs_u_null,
        weigths_ct_pairs_null,
        cell_type_pairs,
        gene_pairs_per_ct_pair_ind,
        M,
    )

    print("Obtaining the communication results...")

    cellcom_results = get_cell_communication_results(
        counts,
        results,
        gene_pairs_per_ct_pair,
        gene_pairs_per_ct_pair_ind,
        cell_types.tolist(),
        cell_type_pairs,
        gene_pairs_ind,
        D,
        test='non-parametric',
    )

    adata.uns["cell_communication_df"] = cellcom_results[0]
    adata.uns["lcs_3d"] = cellcom_results[1]
    adata.uns["gene_pairs_ind_new"] = cellcom_results[2]
    adata.uns["cell_type_list"] = cellcom_results[3]
    adata.uns["genes"] = genes


def _map_fun_parallel_pairs_centered_cond_cellcom_p(ct_pair_i):
    global g_genes
    global g_counts
    global g_weights
    global g_eg2s
    global g_cell_types
    global g_cell_type_pairs
    global g_gene_pairs_ind
    global g_gene_pairs_per_ct_pair_ind
    return _compute_hs_pairs_inner_centered_cond_sym_cellcom_p(
        ct_pair_i,
        g_counts,
        g_weights,
        g_eg2s,
        g_cell_types,
        g_cell_type_pairs,
        g_gene_pairs_ind,
        g_gene_pairs_per_ct_pair_ind,
    )


def _map_fun_parallel_pairs_centered_cond_cellcom_np(ct_pair):
    global g_weights
    global g_neighbors
    global g_counts
    global g_cell_types
    global g_gene_pairs_per_ct_pair_ind
    global g_genes
    global g_cells
    global g_M
    return _compute_hs_pairs_inner_centered_cond_sym_cellcom_np(
        ct_pair,
        g_weights,
        g_neighbors,
        g_counts,
        g_cell_types,
        g_gene_pairs_per_ct_pair_ind,
        g_genes,
        g_cells,
        g_M,
    )


def get_interacting_cell_type_pairs(x, weights, cell_types):
    ct_1, ct_2 = x

    ct_1_bin = cell_types == ct_1
    ct_2_bin = cell_types == ct_2

    weights = weights.tocsc()
    cell_types_weights = weights[ct_1_bin,][:, ct_2_bin]
    # cell_types_weights = sparse.COO.from_scipy_sparse(cell_types_weights)

    return bool(cell_types_weights.nnz)


# @jit(nopython=True)
def conditional_eg2_cellcom(ct_pair, counts, weights, cell_types):
    """
    Computes EG2 for the conditional correlation
    """
    ct_t, ct_u = ct_pair
    cell_type_t_mask = [ct == ct_t for ct in cell_types]
    cell_type_u_mask = [ct == ct_u for ct in cell_types]

    weights_ct = weights[cell_type_t_mask,][:, cell_type_u_mask]
    weights_ct_sq = np.power(weights_ct, 2)
    counts_ct_t = counts[:, cell_type_t_mask]
    counts_ct_t_sq = np.power(counts_ct_t, 2)
    eg2_matrix = weights_ct_sq.T @ counts_ct_t_sq.T
    out_eg2 = list(np.sum(eg2_matrix, axis=0))

    return out_eg2


def conditional_eg2_cellcom_fast(counts_ct_pairs_t, counts_ct_pairs_u, weigths_ct_pairs):
    """
    Computes EG2 for the conditional correlation using PyTorch tensors
    """

    weigths_ct_pairs_sq_data = weigths_ct_pairs.data ** 2
    weigths_ct_pairs_sq = sparse.COO(weigths_ct_pairs.coords, weigths_ct_pairs_sq_data, shape=weigths_ct_pairs.shape)

    counts_ct_pairs_t_sq_data = counts_ct_pairs_t.data ** 2
    counts_ct_pairs_t_sq = sparse.COO(counts_ct_pairs_t.coords, counts_ct_pairs_t_sq_data, shape=counts_ct_pairs_t.shape)

    eg2_matrix = sparse.einsum("cai,cij->caj", counts_ct_pairs_t_sq, weigths_ct_pairs_sq)

    out_eg2 = sparse.sum(eg2_matrix, axis=2)


    # weigths_ct_pairs_sq = torch.pow(weigths_ct_pairs, 2)
    # counts_ct_pairs_t_sq = torch.pow(counts_ct_pairs_t, 2)

    # eg2_matrix = torch.einsum("cai,cij->caj", counts_ct_pairs_t_sq, weigths_ct_pairs_sq)

    # out_eg2 = torch.sum(eg2_matrix, dim=2)

    return out_eg2


def conditional_eg2_cellcom_slow(ct_pair, counts, neighbors, weights, cell_types):
    """
    Computes EG2 for the conditional correlation
    """
    ct_t, ct_u = ct_pair
    cell_type_t_mask = cell_types == ct_t
    cell_type_t_mask = cell_type_t_mask.reset_index(drop=True)
    cell_type_t_mask_ind = cell_type_t_mask[cell_type_t_mask].index.tolist()
    cell_type_u_mask = cell_types == ct_u
    cell_type_u_mask = cell_type_u_mask.reset_index(drop=True)
    cell_type_u_mask_ind = cell_type_u_mask[cell_type_u_mask].index.tolist()

    neighbors_ct = neighbors[cell_type_t_mask,]
    weights_ct = weights[cell_type_t_mask,]

    N = neighbors_ct.shape[0]

    t1x_multi = [np.zeros(N) for gene in range(counts.shape[0])]

    for i in range(N):
        neighbors_ct_i = neighbors_ct[i]
        neighbors_ct_i = neighbors_ct_i[~np.isnan(neighbors_ct_i)]
        neighbors_ct_i = [neighbor_ct_i for neighbor_ct_i in neighbors_ct_i if neighbor_ct_i in cell_type_u_mask_ind]
        if len(neighbors_ct_i) == 0:
            continue

        for j in neighbors_ct_i:
            ind_i = cell_type_t_mask_ind[i]
            j = int(j)
            w_ij = weights_ct[i, j]

            for l in range(counts.shape[0]):
                xi = counts[l, ind_i]

                t1x_multi[l][i] += (w_ij * xi) ** 2

    out_eg2 = [t1x.sum() for t1x in t1x_multi]

    return out_eg2


def create_centered_counts_ct(counts, model, num_umi, cell_types):
    """
    Creates a matrix of centered/standardized counts given
    the selected statistical model
    """
    out = np.zeros_like(counts, dtype="double")

    for i in tqdm(range(out.shape[0])):
        vals_x = counts[i]

        out_x = create_centered_counts_row_ct(vals_x, model, num_umi, cell_types)

        out[i] = out_x

    return out


def create_centered_counts(counts, model, num_umi):
    """
    Creates a matrix of centered/standardized counts given
    the selected statistical model
    """
    out = np.zeros_like(counts, dtype="double")

    for i in tqdm(range(out.shape[0])):
        vals_x = counts[i]

        out_x = create_centered_counts_row(vals_x, model, num_umi)

        out[i] = out_x

    return out


def create_centered_counts_row_ct(vals_x, model, num_umi, cell_types):
    if model == "bernoulli":
        vals_x = (vals_x > 0).astype("double")
        mu_x, var_x, x2_x = models.bernoulli_model(vals_x, num_umi)

    elif model == "danb":
        mu_x, var_x, x2_x = models.ct_danb_model(vals_x, num_umi, cell_types)

    elif model == "normal":
        mu_x, var_x, x2_x = models.normal_model(vals_x, num_umi)

    elif model == "none":
        mu_x, var_x, x2_x = models.none_model(vals_x, num_umi)

    else:
        raise Exception("Invalid Model: {}".format(model))

    var_x[var_x == 0] = 1
    out_x = (vals_x - mu_x) / (var_x**0.5)
    out_x[out_x == 0] = 0

    return out_x


def create_centered_counts_row(vals_x, model, num_umi):
    if model == "bernoulli":
        vals_x = (vals_x > 0).astype("double")
        mu_x, var_x, x2_x = models.bernoulli_model(vals_x, num_umi)

    elif model == "danb":
        mu_x, var_x, x2_x = models.danb_model(vals_x, num_umi)

    elif model == "normal":
        mu_x, var_x, x2_x = models.normal_model(vals_x, num_umi)

    elif model == "none":
        mu_x, var_x, x2_x = models.none_model(vals_x, num_umi)

    else:
        raise Exception("Invalid Model: {}".format(model))

    var_x[var_x == 0] = 1
    out_x = (vals_x - mu_x) / (var_x**0.5)
    out_x[out_x == 0] = 0

    return out_x


# @jit(nopython=True)
def _compute_hs_pairs_inner_centered_cond_sym_cellcom_p(
    ct_pair_i, counts, weights, eg2s, cell_types, cell_type_pairs, gene_pairs_ind, gene_pairs_per_ct_pair_ind
):
    """
    This version assumes that the counts have already been modeled
    and centered
    """

    ct_pair = cell_type_pairs[ct_pair_i]
    ct_t, ct_u = ct_pair
    cell_type_t_mask = cell_types == ct_t
    cell_type_t_mask = cell_type_t_mask.reset_index(drop=True)
    cell_type_u_mask = cell_types == ct_u
    cell_type_u_mask = cell_type_u_mask.reset_index(drop=True)

    gene_pairs_ind = gene_pairs_per_ct_pair_ind[
        ct_pair
    ]  # If we consider all the gene pairs (irrespective of the cell type pair) use 'gene_pairs_ind' directly

    weights_ct = weights[cell_type_t_mask,][:, cell_type_u_mask]
    counts_ct_t = counts[:, cell_type_t_mask]
    counts_ct_u = counts[:, cell_type_u_mask]
    gene_pair_autocor = counts_ct_t @ weights_ct @ counts_ct_u.T

    Wtot2 = (weights_ct**2).sum()

    C = []
    EG2 = []
    for gene_pair_ind in gene_pairs_ind:
        g1_ind, g2_ind = gene_pair_ind
        lc = gene_pair_autocor[g1_ind, g2_ind]
        if g1_ind == g2_ind:
            eg2 = Wtot2
        else:
            eg2 = eg2s[ct_pair_i][g1_ind]
        C.append(lc)
        EG2.append(eg2)

    EG = [0 for i in range(len(gene_pairs_ind))]

    stdG = [(EG2[i] - EG[i] ** 2) ** 0.5 for i in range(len(gene_pairs_ind))]
    stdG = [1 if stdG[i] == 0 else stdG[i] for i in range(len(stdG))]

    Z = [(C[i] - EG[i]) / stdG[i] for i in range(len(gene_pairs_ind))]

    return (C, Z)


def compute_cellcom_p(
    counts_ct_pairs_t,
    counts_ct_pairs_u,
    weigths_ct_pairs,
    eg2s,
    cell_types,
    cell_type_pairs,
    gene_pairs_ind,
    gene_pairs_per_ct_pair_ind,
):
    """
    This version assumes that the counts have already been modeled
    and centered
    """

    counts_ct_pairs_u_perm = sparse.permute_dims(counts_ct_pairs_u, [0,2,1])
    gene_pair_cor = sparse.einsum(
        "cat,ctu,cub->cab", counts_ct_pairs_t, weigths_ct_pairs, counts_ct_pairs_u_perm
    )

    weigths_ct_pairs_sq_data = weigths_ct_pairs.data ** 2
    weigths_ct_pairs_sq = sparse.COO(weigths_ct_pairs.coords, weigths_ct_pairs_sq_data, shape=weigths_ct_pairs.shape)
    Wtot2 = sparse.sum(weigths_ct_pairs_sq, axis=(1, 2))

    compute_Z_scores_cellcom_p_partial = partial(
        compute_Z_scores_cellcom_p,
        cell_type_pairs=cell_type_pairs,
        gene_pair_cor=gene_pair_cor,
        gene_pairs_per_ct_pair_ind=gene_pairs_per_ct_pair_ind,
        Wtot2=Wtot2,
        eg2s=eg2s,
    )

    results = list(map(compute_Z_scores_cellcom_p_partial, cell_type_pairs))

    # gene_pair_cor = torch.einsum(
    #     "cat,ctu,cub->cab", counts_ct_pairs_t, weigths_ct_pairs, counts_ct_pairs_u.permute(0, 2, 1)
    # )

    # Wtot2 = torch.sum(torch.pow(weigths_ct_pairs, 2), dim=(1, 2))

    # compute_Z_scores_cellcom_p_partial = partial(
    #     compute_Z_scores_cellcom_p,
    #     cell_type_pairs=cell_type_pairs,
    #     gene_pair_cor=gene_pair_cor,
    #     gene_pairs_per_ct_pair_ind=gene_pairs_per_ct_pair_ind,
    #     Wtot2=Wtot2,
    #     eg2s=eg2s,
    # )

    # results = list(map(compute_Z_scores_cellcom_p_partial, cell_type_pairs))

    return results


def compute_Z_scores_cellcom_p(ct_pair, cell_type_pairs, gene_pair_cor, gene_pairs_per_ct_pair_ind, Wtot2, eg2s):
    i = cell_type_pairs.index(ct_pair)
    gene_pair_cor_ct = gene_pair_cor[i, :, :]
    gene_pairs_ind = gene_pairs_per_ct_pair_ind[
        ct_pair
    ]  # If we consider all the gene pairs (irrespective of the cell type pair) use 'gene_pairs_ind' directly

    C = []
    EG2 = []
    for gene_pair_ind in gene_pairs_ind:
        g1_ind, g2_ind = gene_pair_ind
        lc = gene_pair_cor_ct[g1_ind, g2_ind]
        if g1_ind == g2_ind:
            eg2 = Wtot2[i]
        else:
            eg2 = eg2s[i][g1_ind]
        C.append(lc)
        EG2.append(eg2)

    EG = [0 for i in range(len(gene_pairs_ind))]

    stdG = [(EG2[i] - EG[i] ** 2) ** 0.5 for i in range(len(gene_pairs_ind))]
    stdG = [1 if stdG[i] == 0 else stdG[i] for i in range(len(stdG))]

    Z = [(C[i] - EG[i]) / stdG[i] for i in range(len(gene_pairs_ind))]

    return (C, Z)


# @jit(nopython=True)
def _compute_hs_pairs_inner_centered_cond_sym_cellcom_np(
    ct_pair, weights, neighbors, counts, cell_types, gene_pairs_per_ct_pair_ind, genes, cells, M
):
    ct_t, ct_u = ct_pair
    cell_type_t_mask = cell_types == ct_t
    cell_type_t_mask = cell_type_t_mask.reset_index(drop=True)
    cell_type_t_mask_ind = cell_type_t_mask[cell_type_t_mask].index.tolist()
    cell_type_u_mask = cell_types == ct_u
    cell_type_u_mask = cell_type_u_mask.reset_index(drop=True)
    cell_type_u_mask_ind = cell_type_u_mask[cell_type_u_mask].index.tolist()

    neighbors_ct = neighbors[cell_type_t_mask,]
    weights_ct = weights[cell_type_t_mask,]

    gene_pairs_ind = gene_pairs_per_ct_pair_ind[ct_pair]

    # interacting_cells = {}

    # for gp_ind in gene_pairs_ind:

    #     gene_x_ind, gene_y_ind = gp_ind

    #     for i in range(neighbors_ct.shape[0]):

    #         neighbors_ct_i = neighbors_ct[i]
    #         neighbors_ct_i = neighbors_ct_i[~np.isnan(neighbors_ct_i)]
    #         neighbors_ct_i = [neighbor_ct_i for neighbor_ct_i in neighbors_ct_i if neighbor_ct_i in cell_type_u_mask_ind]
    #         if len(neighbors_ct_i) == 0:
    #             continue

    #         for j in neighbors_ct_i:

    #             ind_i = cell_type_t_mask_ind[i]
    #             j = int(j)
    #             w_ij = weights_ct[i, j]

    #             xi = counts[gene_x_ind, ind_i]
    #             cell_i = cells[ind_i]
    #             yj = counts[gene_y_ind, j]
    #             cell_j = cells[j]
    #             cell_pair = cell_i + '/' + cell_j

    #             gene_x = genes[gene_x_ind]
    #             gene_y = genes[gene_y_ind]
    #             gene_pair = gene_x + '_' + gene_y

    #             if not xi == 0 or yj == 0 or w_ij == 0:
    #                 if gene_pair not in interacting_cells.keys():
    #                     interacting_cells[gene_pair] = {"cells": [], "scores": []}
    #                 interacting_cells[gene_pair]["cells"].append(cell_pair)
    #                 interacting_cells[gene_pair]["scores"].append(w_ij*xi*yj)

    weights_ct = weights[cell_type_t_mask,][:, cell_type_u_mask]
    counts_ct_t = counts[:, cell_type_t_mask]
    counts_ct_u = counts[:, cell_type_u_mask]
    gene_pair_autocor = counts_ct_t @ weights_ct @ counts_ct_u.T

    # cell_type_ind = cell_type_list.index(cell_type)

    # G_max = compute_local_cov_max_ct(counts_ct, D[cell_type_ind])
    # C = gene_pair_autocor.diagonal() / G_max

    C = []
    for gene_pair_ind in gene_pairs_ind:
        g1_ind, g2_ind = gene_pair_ind
        C.append(gene_pair_autocor[g1_ind, g2_ind])

    # Compute null distribution using permutations
    print("Computing null distribution for {}...".format(ct_pair))

    gene_pair_autocor_null = []
    for i in tqdm(range(M)):
        shuffled_cell_types = sorted(cell_types.tolist(), key=lambda x: random())
        cell_type_t_mask = [ct == ct_t for ct in shuffled_cell_types]
        cell_type_u_mask = [ct == ct_u for ct in shuffled_cell_types]

        weights_ct = weights[cell_type_t_mask,][:, cell_type_u_mask]
        counts_ct_t = counts[:, cell_type_t_mask]
        counts_ct_u = counts[:, cell_type_u_mask]
        gene_pair_autocor_rand = counts_ct_t @ weights_ct @ counts_ct_u.T

        gene_pair_autocor_rand_list = []
        for gene_pair_ind in gene_pairs_ind:
            g1_ind, g2_ind = gene_pair_ind
            gene_pair_autocor_rand_list.append(gene_pair_autocor_rand[g1_ind, g2_ind])
        gene_pair_autocor_null.append(gene_pair_autocor_rand_list)

    gene_pair_autocor_null = np.stack(gene_pair_autocor_null, axis=0)
    x = np.sum(C < gene_pair_autocor_null, axis=0)
    p_values = (x + 1) / (M + 1)

    # return [C, p_values, interacting_cells]
    return [C, p_values]


def compute_cellcom_np(
    counts_ct_pairs_t,
    counts_ct_pairs_u,
    weigths_ct_pairs,
    counts_ct_pairs_t_null,
    counts_ct_pairs_u_null,
    weigths_ct_pairs_null,
    cell_type_pairs,
    gene_pairs_per_ct_pair_ind,
    M,
):

    counts_ct_pairs_u_perm = sparse.permute_dims(counts_ct_pairs_u, [0,2,1])
    gene_pair_cor = sparse.einsum(
        "cat,ctu,cub->cab", counts_ct_pairs_t, weigths_ct_pairs, counts_ct_pairs_u_perm
    )

    counts_ct_pairs_u_null_perm = sparse.permute_dims(counts_ct_pairs_u_null, [0,1,3,2])
    gene_pair_cor_null = sparse.einsum(
        "rcat,rctu,rcub->rcab",
        counts_ct_pairs_t_null,
        weigths_ct_pairs_null,
        counts_ct_pairs_u_null_perm,
    )

    gene_pair_cor_expanded = gene_pair_cor.reshape((1,) + gene_pair_cor.shape)
    x = np.sum(gene_pair_cor_null > gene_pair_cor_expanded, axis=0)
    pvals = (x + 1) / (M + 1)

    extract_results_cellcom_np_partial = partial(
        extract_results_cellcom_np,
        cell_type_pairs=cell_type_pairs,
        gene_pair_cor=gene_pair_cor,
        pvals=pvals,
        gene_pairs_per_ct_pair_ind=gene_pairs_per_ct_pair_ind,
    )
    results = list(map(extract_results_cellcom_np_partial, cell_type_pairs))


    return results


def extract_results_cellcom_np(ct_pair, cell_type_pairs, gene_pair_cor, pvals, gene_pairs_per_ct_pair_ind):
    i = cell_type_pairs.index(ct_pair)
    gene_pair_cor_ct = gene_pair_cor[i, :, :]
    pvals_ct = pvals[i, :, :]
    gene_pairs_ind = gene_pairs_per_ct_pair_ind[
        ct_pair
    ]  # If we consider all the gene pairs (irrespective of the cell type pair) use 'gene_pairs_ind' directly

    C = []
    p_values = []
    for gene_pair_ind in gene_pairs_ind:
        g1_ind, g2_ind = gene_pair_ind
        lc = gene_pair_cor_ct[g1_ind, g2_ind]
        p_value = pvals_ct[g1_ind, g2_ind]

        C.append(lc.reshape(1))
        p_values.append(p_value.reshape(1))

    C = list(np.concatenate(C))
    p_values = list(np.concatenate(p_values))

    return (C, p_values)


# @njit
def expand_ct_pairs_cellcom(pairs, vals, N):
    out = [np.zeros((N, N)) for k in range(len(vals))]

    for k in range(len(out)):
        for i in range(len(pairs)):
            x = pairs[i, 0]
            y = pairs[i, 1]
            v = vals[k][i]

            out[k][x, y] = v

    return out


def expand_gene_pairs_cellcom(pairs, ct_pairs, cell_types, vals, N):

    pairs_new = []
    for pair in pairs:
        p1, p2 = pair
        if ([p1, p2] not in pairs_new) and ([p2, p1] not in pairs_new):
            pairs_new.append([p1, p2])

    out = [np.zeros((N, N)) for k in range(len(pairs_new))]

    for k in range(len(out)):
        p1, p2 = pairs[k]
        for i in range(len(ct_pairs)):
            ct_1, ct_2 = ct_pairs[i]
            x = np.where(cell_types == ct_1)[0]
            y = np.where(cell_types == ct_2)[0]
            if p1 == p2:
                j = np.where((pairs[:, 0] == p1) & (pairs[:, 1] == p2))[0]
                v = vals[i][j]
                out[k][x, y] = v
            else:
                j1 = np.where((pairs[:, 0] == p1) & (pairs[:, 1] == p2))[0]
                j2 = np.where((pairs[:, 0] == p2) & (pairs[:, 1] == p1))[0]
                v1 = vals[i][j1]
                v2 = vals[i][j2]
                out[k][x, y] = v1
                out[k][y, x] = v2

    return out, pairs_new


@njit
def expand_pairs(pairs, vals, N):
    out = np.zeros((N, N))

    for i in range(len(pairs)):
        x = pairs[i, 0]
        y = pairs[i, 1]
        v = vals[i]

        out[x, y] = v
        out[y, x] = v

    return out


def compute_local_cov_pairs_max(node_degrees, counts):
    """
    For a Genes x Cells count matrix, compute the maximal pair-wise correlation
    between any two genes
    """

    N_GENES = counts.shape[0]

    gene_maxs = np.zeros(N_GENES)
    for i in range(N_GENES):
        gene_maxs[i] = compute_local_cov_max(counts[i].todense(), node_degrees)

    result = gene_maxs.reshape((-1, 1)) + gene_maxs.reshape((1, -1))
    result = result / 2
    return result


def get_ct_pair_counts_and_weights(counts, weights, cell_type_pairs, cell_types, gene_pairs_per_ct_pair_ind):

    # counts_ct_pairs_t = sparse.stack([counts for _ in range(depth)], axis=0)
    # weigths_ct_pairs = sparse.stack([weights for _ in range(depth)], axis=0)

    # counts = torch.stack([torch.tensor(counts)] * len(cell_type_pairs), dim=0) # We create a 3D tensor where the counts matrix is repeated len(cell_type_pairs) times
    # weigths_ct_pairs = torch.stack([torch.tensor(weights)] * len(cell_type_pairs), dim=0)

    # counts_ct_pairs_t = counts.clone()
    # counts_ct_pairs_u = counts.clone()

    # n_cell_type_pairs = len(cell_type_pairs)



    # counts_ct_pairs_t, counts_ct_pairs_u = get_ct_pair_counts(
    #     counts, cell_type_pairs, cell_types, gene_pairs_per_ct_pair_ind
    # )
    # weigths_ct_pairs = get_ct_pair_weights(weights, cell_type_pairs, cell_types)



    c_nrow, c_ncol = counts.shape
    w_nrow, w_ncol = weights.shape
    n_ct_pairs = len(cell_type_pairs)

    extract_counts_weights_results = partial(
        extract_ct_pair_counts_weights,
        counts=counts,
        weights=weights,
        cell_type_pairs=cell_type_pairs,
        cell_types=cell_types,
        gene_pairs_per_ct_pair_ind=gene_pairs_per_ct_pair_ind,
    )
    results = list(map(extract_counts_weights_results, cell_type_pairs))

    c_new_data_t_all = [x[0] for x in results]
    c_new_coords_3d_t_all = [x[1] for x in results]
    c_new_coords_3d_t_all = np.hstack(c_new_coords_3d_t_all)
    c_new_data_t_all = np.concatenate(c_new_data_t_all)

    c_new_data_u_all = [x[2] for x in results]
    c_new_coords_3d_u_all = [x[3] for x in results]
    c_new_coords_3d_u_all = np.hstack(c_new_coords_3d_u_all)
    c_new_data_u_all = np.concatenate(c_new_data_u_all)

    w_new_data_all = [x[4] for x in results]
    w_new_coords_3d_all = [x[5] for x in results]
    w_new_coords_3d_all = np.hstack(w_new_coords_3d_all)
    w_new_data_all = np.concatenate(w_new_data_all)

    counts_ct_pairs_t = sparse.COO(c_new_coords_3d_t_all, c_new_data_t_all, shape=(n_ct_pairs, c_nrow, c_ncol))
    counts_ct_pairs_u = sparse.COO(c_new_coords_3d_u_all, c_new_data_u_all, shape=(n_ct_pairs, c_nrow, c_ncol))
    weigths_ct_pairs = sparse.COO(w_new_coords_3d_all, w_new_data_all, shape=(n_ct_pairs, w_nrow, w_ncol))


    return counts_ct_pairs_t, counts_ct_pairs_u, weigths_ct_pairs


def get_ct_pair_counts_and_weights_null(counts_ct_pairs_t, counts_ct_pairs_u, weights, cell_type_pairs, cell_types, M):

    n_cells = counts_ct_pairs_t.shape[2]
    cell_permutations = np.vstack([np.random.permutation(n_cells) for _ in range(M)])

    n_ct_pairs, c_nrow, c_ncol = counts_ct_pairs_t.shape
    w_nrow, w_ncol = weights.shape

    extract_counts_weights_results_null = partial(
        extract_ct_pair_counts_weights_null,
        permutations=cell_permutations,
        counts_ct_pairs_t=counts_ct_pairs_t,
        counts_ct_pairs_u=counts_ct_pairs_u,
        weights=weights,
        cell_type_pairs=cell_type_pairs,
        cell_types=cell_types,
    )
    results_null = list(map(extract_counts_weights_results_null, cell_permutations))

    c_null_data_t_all = [x[0] for x in results_null]
    c_null_coords_4d_t_all = [x[1] for x in results_null]
    c_null_coords_4d_t_all = np.hstack(c_null_coords_4d_t_all)
    c_null_data_t_all = np.concatenate(c_null_data_t_all)

    c_null_data_u_all = [x[2] for x in results_null]
    c_null_coords_4d_u_all = [x[3] for x in results_null]
    c_null_coords_4d_u_all = np.hstack(c_null_coords_4d_u_all)
    c_null_data_u_all = np.concatenate(c_null_data_u_all)

    w_null_data_all = [x[4] for x in results_null]
    w_null_coords_4d_all = [x[5] for x in results_null]
    w_null_coords_4d_all = np.hstack(w_null_coords_4d_all)
    w_null_data_all = np.concatenate(w_null_data_all)

    counts_ct_pairs_t_null = sparse.COO(c_null_coords_4d_t_all, c_null_data_t_all, shape=(M, n_ct_pairs, c_nrow, c_ncol))
    counts_ct_pairs_u_null = sparse.COO(c_null_coords_4d_u_all, c_null_data_u_all, shape=(M, n_ct_pairs, c_nrow, c_ncol))
    weigths_ct_pairs_null = sparse.COO(w_null_coords_4d_all, w_null_data_all, shape=(M, n_ct_pairs, w_nrow, w_ncol))


    return counts_ct_pairs_t_null, counts_ct_pairs_u_null, weigths_ct_pairs_null


def extract_ct_pair_counts_weights(ct_pair, counts, weights, cell_type_pairs, cell_types, gene_pairs_per_ct_pair_ind):

    i = cell_type_pairs.index(ct_pair)

    ct_t, ct_u = cell_type_pairs[i]
    gene_pairs_per_ct_pair_ind_i = gene_pairs_per_ct_pair_ind[(ct_t, ct_u)]
    ct_t_mask = cell_types.values == ct_t
    ct_t_mask_coords = np.argwhere(ct_t_mask)
    ct_u_mask = cell_types.values == ct_u
    ct_u_mask_coords = np.argwhere(ct_u_mask)

    ct_t_genes = np.unique([t[0] for t in gene_pairs_per_ct_pair_ind_i])
    ct_u_genes = np.unique([t[1] for t in gene_pairs_per_ct_pair_ind_i])

    c_old_coords = counts.coords
    w_old_coords = weights.coords

    # Counts

    c_row_coords_t, c_col_coords_t = np.meshgrid(ct_t_genes, ct_t_mask_coords, indexing='ij')
    c_row_coords_t = c_row_coords_t.ravel()
    c_col_coords_t = c_col_coords_t.ravel()
    c_new_coords_t = np.vstack((c_row_coords_t, c_col_coords_t))

    c_row_coords_u, c_col_coords_u = np.meshgrid(ct_u_genes, ct_u_mask_coords, indexing='ij')
    c_row_coords_u = c_row_coords_u.ravel()
    c_col_coords_u = c_col_coords_u.ravel()
    c_new_coords_u = np.vstack((c_row_coords_u, c_col_coords_u))

    c_matching_indices_t = np.where(np.all(np.isin(c_old_coords.T, c_new_coords_t.T), axis=1))[0]
    c_new_data_t = counts.data[c_matching_indices_t]
    c_new_coords_t = c_old_coords[:,c_matching_indices_t]

    c_matching_indices_u = np.where(np.all(np.isin(c_old_coords.T, c_new_coords_u.T), axis=1))[0]
    c_new_data_u = counts.data[c_matching_indices_u]
    c_new_coords_u = c_old_coords[:,c_matching_indices_u]

    c_coord_3d_t = np.full(c_new_coords_t.shape[1], fill_value=i)
    c_new_coords_3d_t = np.vstack((c_coord_3d_t, c_new_coords_t))

    c_coord_3d_u = np.full(c_new_coords_u.shape[1], fill_value=i)
    c_new_coords_3d_u = np.vstack((c_coord_3d_u, c_new_coords_u))

    # Weights

    w_row_coords, w_col_coords = np.meshgrid(ct_t_mask_coords, ct_u_mask_coords, indexing='ij')
    w_row_coords = w_row_coords.ravel()
    w_col_coords = w_col_coords.ravel()
    w_new_coords = np.vstack((w_row_coords, w_col_coords))

    w_matching_indices = np.where(np.all(np.isin(w_old_coords.T, w_new_coords.T), axis=1))[0]
    w_new_data = weights.data[w_matching_indices]
    w_new_coords = w_old_coords[:,w_matching_indices]

    w_coord_3d = np.full(w_new_coords.shape[1], fill_value=i)
    w_new_coords_3d = np.vstack((w_coord_3d, w_new_coords))


    return (c_new_data_t, c_new_coords_3d_t, c_new_data_u, c_new_coords_3d_u, w_new_data, w_new_coords_3d)


def extract_ct_pair_weights(ct_pair, weights, cell_type_pairs, cell_types):

    i = cell_type_pairs.index(ct_pair)

    ct_t, ct_u = cell_type_pairs[i]
    ct_t_mask = cell_types.values == ct_t
    ct_t_mask_coords = np.argwhere(ct_t_mask)
    ct_u_mask = cell_types.values == ct_u
    ct_u_mask_coords = np.argwhere(ct_u_mask)

    w_old_coords = weights.coords

    w_row_coords, w_col_coords = np.meshgrid(ct_t_mask_coords, ct_u_mask_coords, indexing='ij')
    w_row_coords = w_row_coords.ravel()
    w_col_coords = w_col_coords.ravel()
    w_new_coords = np.vstack((w_row_coords, w_col_coords))

    w_matching_indices = np.where(np.all(np.isin(w_old_coords.T, w_new_coords.T), axis=1))[0]
    w_new_data = weights.data[w_matching_indices]
    w_new_coords = w_old_coords[:,w_matching_indices]

    w_coord_3d = np.full(w_new_coords.shape[1], fill_value=i)
    w_new_coords_3d = np.vstack((w_coord_3d, w_new_coords))


    return (w_new_data, w_new_coords_3d)


def extract_ct_pair_counts_weights_null(permutation, permutations, counts_ct_pairs_t, counts_ct_pairs_u,  weights, cell_type_pairs, cell_types):

    i = np.where(np.all(permutations == permutation, axis=1))[0]
    cell_types_perm = pd.Series(cell_types[permutation])

    counts_ct_pairs_t_perm = counts_ct_pairs_t[:, :, permutation]
    c_perm_coords_t = counts_ct_pairs_t_perm.coords
    c_perm_data_t = counts_ct_pairs_t_perm.data
    counts_ct_pairs_u_perm = counts_ct_pairs_u[:, :, permutation]
    c_perm_coords_u = counts_ct_pairs_u_perm.coords
    c_perm_data_u = counts_ct_pairs_u_perm.data

    extract_weights_results = partial(
        extract_ct_pair_weights,
        weights=weights,
        cell_type_pairs=cell_type_pairs,
        cell_types=cell_types_perm
    )
    weights_results = list(map(extract_weights_results, cell_type_pairs))

    w_perm_data_all = [x[0] for x in weights_results]
    w_perm_coords_3d_all = [x[1] for x in weights_results]
    w_perm_coords_3d_all = np.hstack(w_perm_coords_3d_all)
    w_perm_data_all = np.concatenate(w_perm_data_all)

    c_coord_4d_t = np.full(c_perm_coords_t.shape[1], fill_value=i)
    c_perm_coords_4d_t = np.vstack((c_coord_4d_t, c_perm_coords_t))

    c_coord_4d_u = np.full(c_perm_coords_u.shape[1], fill_value=i)
    c_perm_coords_4d_u = np.vstack((c_coord_4d_u, c_perm_coords_u))

    w_coord_4d_u = np.full(w_perm_coords_3d_all.shape[1], fill_value=i)
    w_perm_coords_4d_u = np.vstack((w_coord_4d_u, w_perm_coords_3d_all))


    return (c_perm_data_t, c_perm_coords_4d_t, c_perm_data_u, c_perm_coords_4d_u, w_perm_data_all, w_perm_coords_4d_u)


def get_cell_communication_results(
    counts, results, gene_pairs_per_ct_pair, gene_pairs_per_ct_pair_ind, cell_types, cell_type_pairs, pairs, D, test,
):
    cell_com_df = pd.DataFrame.from_dict(gene_pairs_per_ct_pair, orient="index").stack().to_frame().reset_index()
    cell_com_df = cell_com_df.drop(["level_1"], axis=1)
    cell_com_df = cell_com_df.rename(columns={"level_0": "cell_type_pair", 0: "gene_pair"})
    cell_com_df["Cell Type 1"], cell_com_df["Cell Type 2"] = zip(*cell_com_df["cell_type_pair"])
    cell_com_df["Gene 1"], cell_com_df["Gene 2"] = zip(*cell_com_df["gene_pair"])

    N = counts.shape[0]
    pairs_array = np.array(pairs)

    vals_lc = [x[0] for x in results]
    if test == 'parametric':
        z_values = [x[1] for x in results]
    else:
        p_values = [x[1] for x in results]

    lcs = expand_ct_pairs_cellcom(pairs_array, np.array(vals_lc), N)
    if test == 'parametric':
        lc_zs = expand_ct_pairs_cellcom(pairs_array, np.array(z_values), N)

    lc_values = []

    for i in range(len(cell_type_pairs)):

        ct_pair = cell_type_pairs[i]
        ct_t, ct_u = ct_pair
        cell_type_t_mask = [ct == ct_t for ct in cell_types]
        counts_ct_t = counts[:,cell_type_t_mask]
        D_ct_t = D[i][cell_type_t_mask]
        lc_maxs = compute_local_cov_pairs_max(D_ct_t, counts_ct_t)
        lcs[i] = lcs[i] / lc_maxs

        gene_pairs_ind = gene_pairs_per_ct_pair_ind[ct_pair]

        for gene_pair_ind in gene_pairs_ind:
            g1_ind, g2_ind = gene_pair_ind
            lc_values.append(lcs[i][g1_ind, g2_ind])

    cell_type_list = np.unique(cell_types)
    n_cell_types = len(cell_type_list)
    lcs, gene_pairs_ind_new = expand_gene_pairs_cellcom(pairs_array, cell_type_pairs, cell_type_list, np.array(vals_lc), n_cell_types)
    lcs_3d = np.stack(lcs)
    if test == 'parametric':
        lc_zs, gene_pairs_ind_new = expand_gene_pairs_cellcom(pairs_array, cell_type_pairs, cell_type_list, np.array(z_values), n_cell_types)
        lc_zs_3d = np.stack(lc_zs)

    vals_lc = [item.item() for sublist in vals_lc for item in sublist]
    if test == 'parametric':
        z_values = [item.item() for sublist in z_values for item in sublist]
    else:
        p_values = [item.item() for sublist in p_values for item in sublist]

    cell_com_df["C"] = vals_lc
    cell_com_df['C_norm'] = lc_values
    if test == 'parametric':
        cell_com_df["Z"] = z_values
        cell_com_df["Pval"] = norm.sf(cell_com_df["Z"].values)
    else:
        cell_com_df["Pval"] = p_values

    cell_communication_df = cell_com_df[["Cell Type 1", "Cell Type 2", "Gene 1", "Gene 2", "C", "C_norm", "Z", "Pval"]] if test == 'parametric' else cell_com_df[["Cell Type 1", "Cell Type 2", "Gene 1", "Gene 2", "C", "C_norm", "Pval"]]
    cell_communication_df["FDR"] = multipletests(cell_communication_df["Pval"], method="fdr_bh")[1]

    if test == 'parametric':
        return (cell_communication_df, lcs_3d, lc_zs_3d, gene_pairs_ind_new, cell_type_list)
    else:
        return (cell_communication_df, lcs_3d, gene_pairs_ind_new, cell_type_list)


def process_communication_output(
        adata: AnnData,
        cell_communication_df: pd.DataFrame,
        m2h_info: Optional[dict] = '',
        transporter_info: Optional[pd.DataFrame] = '',
        pathway_dict: Optional[dict] = None,
        org: Optional[Union[Literal["Mouse"], Literal["Human"]]] = "Mouse",
    ):
    """Compute gene pair and metabolite scores for each cell.

    Parameters
    ----------
    adata
        AnnData object to compute communication scores for.

    """

    transporter_df = pd.read_csv(transporter_info, index_col=0) if type(transporter_info) == str else transporter_info

    if type(m2h_info) == str:
        with open(m2h_info) as json_file:
            m2h_dict = json.load(json_file)
    else:
        m2h_dict = m2h_info

    cell_communication_df["cell_type_pair"] = cell_communication_df.apply(lambda row: (row['Cell Type 1'], row['Cell Type 2']), axis=1)
    cell_communication_df["gene_pair"] = cell_communication_df.apply(lambda row: (row['Gene 1'], row['Gene 2']), axis=1)

    cell_communication_df['cells'] = cell_communication_df['interacting_cells'].apply(lambda x: x['cells'])
    cell_communication_df['scores'] = cell_communication_df['interacting_cells'].apply(lambda x: x['scores'])

    gene_pairs_per_metabolite = adata.uns["gene_pairs_per_metabolite"]

    metabolite_gene_pair_df = pd.DataFrame.from_dict(gene_pairs_per_metabolite, orient="index").reset_index()
    metabolite_gene_pair_df = metabolite_gene_pair_df.rename(columns={"index": "metabolite"})

    metabolite_gene_pair_df['gene_pair'] = metabolite_gene_pair_df['gene_pair'].apply(lambda arr: [(sub_array[0], sub_array[1]) for sub_array in arr])
    metabolite_gene_pair_df['gene_type'] = metabolite_gene_pair_df['gene_type'].apply(lambda arr: [(sub_array[0], sub_array[1]) for sub_array in arr])

    metabolite_gene_pair_df = pd.concat(
        [
            metabolite_gene_pair_df['metabolite'],
            metabolite_gene_pair_df.explode('gene_pair')['gene_pair'],
            metabolite_gene_pair_df.explode('gene_type')['gene_type'],
        ],
        axis=1,
    )
    metabolite_gene_pair_df = metabolite_gene_pair_df.reset_index(drop=True)

    metabolite_gene_pair_df = metabolite_gene_pair_df.groupby(["gene_pair", "gene_type"])[
        "metabolite"
    ].apply(tuple).reset_index()
    metabolite_gene_pair_df = metabolite_gene_pair_df.drop_duplicates()

    ct_pair_gene_pair_metab_df = cell_communication_df.merge(
        metabolite_gene_pair_df, on="gene_pair", validate="many_to_many"
    )

    if pathway_dict is not None:
        for pathway_id in pathway_dict.keys():
            ct_pair_gene_pair_metab_df[pathway_dict[pathway_id]] = ct_pair_gene_pair_metab_df.gene_pair.apply(get_metabolic_pathway, m2h_dict=m2h_dict, transporter_df=transporter_df, org=org, col=pathway_id)


    return ct_pair_gene_pair_metab_df


def compute_tensor_factorization(adata, rank=5, z_scores=True):

    gene_pairs_ind = adata.uns["gene_pairs_ind_new"]
    cell_types = adata.uns["cell_type_list"]
    genes = adata.uns["genes"]

    tensor_3d = adata.uns['lc_zs_3d'] if z_scores is True else adata.uns['lcs_3d']

    factors = tl.decomposition.parafac(tensor_3d, rank=rank)
    factor_names = [f'Factor {i+1}' for i in range(rank)]
    gene_pairs = [(genes[gp1], genes[gp2]) for gp1, gp2 in gene_pairs_ind]

    gene_pair_factors = pd.DataFrame(factors.factors[0], index=gene_pairs, columns=factor_names)
    ct_1_factors = pd.DataFrame(factors.factors[1], index=cell_types, columns=factor_names)
    ct_2_factors = pd.DataFrame(factors.factors[2], index=cell_types, columns=factor_names)

    return (gene_pair_factors, ct_1_factors, ct_2_factors)


def compute_communication_scores(
        adata: AnnData,
        ct_pair_gene_pair_metab_df: pd.DataFrame,
        layer_key: Optional[Union[Literal["use_raw"], str]] = None,
        var: Optional[list] = None,
        var_subset: Optional[Union[list, dict]] = None,
        per_ct: Optional[bool] = None,
        per_ct_pair: Optional[bool] = None,
    ):
    """Compute gene pair and metabolite scores for each cell.

    Parameters
    ----------
    adata
        AnnData object to compute communication scores for.

    """
    default_vars = ['gene_pair', 'cell_type', 'cell_type_pair']
    var = list(np.unique(var + default_vars)) if var is not None else default_vars

    ct_pair_gene_pair_metab_df = pd.concat(
        [
            ct_pair_gene_pair_metab_df.drop(['cells', 'scores'] ,axis=1),
            ct_pair_gene_pair_metab_df.explode('cells')['cells'],
            ct_pair_gene_pair_metab_df.explode('scores')['scores'],
        ],
        axis=1,
    )
    ct_pair_gene_pair_metab_df['gene'] = ct_pair_gene_pair_metab_df['gene_pair'].copy()
    ct_pair_gene_pair_metab_df = ct_pair_gene_pair_metab_df.explode(['gene', 'cells']).reset_index(drop=True)

    use_raw = layer_key == "use_raw"
    index = adata.raw.obs.index if use_raw else adata.obs_names

    genes = ct_pair_gene_pair_metab_df['gene'].unique()

    gene_scores_df = pd.DataFrame(index=index, columns=genes, data=np.zeros((len(index), len(genes))))

    for gene in tqdm(genes):
        ct_pair_gene_pair_metab_df_g = ct_pair_gene_pair_metab_df[
            ct_pair_gene_pair_metab_df.gene == gene
        ].copy()

        gene_scores_tmp = ct_pair_gene_pair_metab_df_g.groupby(["cells", "gene_pair"])["scores"].sum()
        gene_scores_df[gene] = gene_scores_tmp.groupby('cells').mean()

        # Cell type pairs

        if per_ct_pair:
            cell_type_pairs = ct_pair_gene_pair_metab_df_g.cell_type_pair.unique()
            cell_type_pairs_str = [ct1 + '_' + ct2 for ct1, ct2 in cell_type_pairs]
            gene_ct_pair_scores_df = pd.DataFrame(
                index=index, columns=cell_type_pairs_str, data=np.zeros((len(index), len(cell_type_pairs_str)))
            )
            gene_ct_pair_scores_df_tmp = (
                ct_pair_gene_pair_metab_df_g.groupby(["cells", "cell_type_pair", "gene_pair"])["scores"].sum().groupby(['cells', "cell_type_pair"]).mean().unstack(level=1)
            )
            for ct_pair in cell_type_pairs:
                ct1, ct2 = ct_pair
                gene_ct_pair_scores_df[ct1 + '_' + ct2] = gene_ct_pair_scores_df_tmp[ct_pair]
            adata.obsm[f"{gene}_ct_pair_scores"] = gene_ct_pair_scores_df.fillna(0)

        # Cell types

        if per_ct:
            ct_pair_gene_pair_metab_df_g = ct_pair_gene_pair_metab_df_g.explode('cell_type_pair').reset_index(drop=True)
            ct_pair_gene_pair_metab_df_g = ct_pair_gene_pair_metab_df_g.rename(columns={'cell_type_pair': 'cell_type'})
            cell_types = ct_pair_gene_pair_metab_df_g.cell_type.unique()
            gene_ct_scores_df = pd.DataFrame(
                index=index, columns=cell_types, data=np.zeros((len(index), len(cell_types)))
            )
            gene_ct_scores_df_tmp = (
                ct_pair_gene_pair_metab_df_g.groupby(["cells", "cell_type", "gene_pair"])["scores"].sum().groupby(['cells', "cell_type"]).mean().unstack(level=1)
            )
            for ct in cell_types:
                gene_ct_scores_df[ct] = gene_ct_scores_df_tmp[ct]
            adata.obsm[f"{gene}_ct_scores"] = gene_ct_scores_df.fillna(0)

    adata.obsm["gene_scores"] = gene_scores_df.fillna(0)


    if 'gene_pair' in var:
        gene_pairs = ct_pair_gene_pair_metab_df.gene_pair.unique()

        gene_pairs_dict = {}
        for gene_pair in gene_pairs:
            pair1, pair2 = gene_pair
            if (pair1 + '_' + pair2 in gene_pairs_dict.keys()):
                gene_pairs_dict[pair1 + '_' + pair2].append(gene_pair)
            elif (pair2 + '_' + pair1 in gene_pairs_dict.keys()):
                gene_pairs_dict[pair2 + '_' + pair1].append(gene_pair)
            else:
                gene_pairs_dict[pair1 + '_' + pair2] = []
                gene_pairs_dict[pair1 + '_' + pair2].append(gene_pair)

        gene_pair_scores_df = pd.DataFrame(index=index, columns=gene_pairs_dict.keys(), data=np.zeros((len(index), len(gene_pairs_dict.keys()))))

        for gene_pair in tqdm(gene_pairs_dict.keys()):
            ct_pair_gene_pair_metab_df_gp = ct_pair_gene_pair_metab_df[
                ct_pair_gene_pair_metab_df.gene_pair.isin(gene_pairs_dict[gene_pair])
            ].copy()

            gene_pair_scores_df[gene_pair] = ct_pair_gene_pair_metab_df_gp.groupby("cells")["scores"].sum()

            # Cell type pairs

            if per_ct_pair:
                cell_type_pairs = ct_pair_gene_pair_metab_df_gp.cell_type_pair.unique()
                cell_type_pairs_str = [ct1 + '_' + ct2 for ct1, ct2 in cell_type_pairs]
                gene_pair_ct_pair_scores_df = pd.DataFrame(
                    index=index, columns=cell_type_pairs_str, data=np.zeros((len(index), len(cell_type_pairs_str)))
                )
                gene_pair_ct_pair_scores_df_tmp = (
                    ct_pair_gene_pair_metab_df_gp.groupby(["cells", "cell_type_pair"])["scores"].sum().unstack(level=1)
                )
                for ct_pair in cell_type_pairs:
                    ct1, ct2 = ct_pair
                    gene_pair_ct_pair_scores_df[ct1 + '_' + ct2] = gene_pair_ct_pair_scores_df_tmp[ct_pair]
                adata.obsm[f"{gene_pair}_ct_pair_scores"] = gene_pair_ct_pair_scores_df.fillna(0)

            # Cell types

            if per_ct:
                ct_pair_gene_pair_metab_df_gp = ct_pair_gene_pair_metab_df_gp.explode('cell_type_pair').reset_index(drop=True)
                ct_pair_gene_pair_metab_df_gp = ct_pair_gene_pair_metab_df_gp.rename(columns={'cell_type_pair': 'cell_type'})
                cell_types = ct_pair_gene_pair_metab_df_gp.cell_type.unique()
                gene_pair_ct_scores_df = pd.DataFrame(
                    index=index, columns=cell_types, data=np.zeros((len(index), len(cell_types)))
                )
                gene_pair_ct_scores_df_tmp = (
                    ct_pair_gene_pair_metab_df_gp.groupby(["cells", "cell_type"])["scores"].sum().unstack(level=1)
                )
                for ct in cell_types:
                    gene_pair_ct_scores_df[ct] = gene_pair_ct_scores_df_tmp[ct]
                adata.obsm[f"{gene_pair}_ct_scores"] = gene_pair_ct_scores_df.fillna(0)

        adata.obsm["gene_pair_scores"] = gene_pair_scores_df.fillna(0)


    var = [v for v in var if v not in default_vars]

    for v in var:
        print(f"Computing scores for the '{v}' variable...")

        ct_pair_gene_pair_metab_df_v = ct_pair_gene_pair_metab_df.explode(v).reset_index(drop=True)
        if len(var) == 1 and type(var_subset) is list:
            ct_pair_gene_pair_metab_df_v = ct_pair_gene_pair_metab_df_v[ct_pair_gene_pair_metab_df_v[v].isin(var_subset)]
        elif len(var) > 1 and type(var_subset) is dict and v in var_subset.keys():
            ct_pair_gene_pair_metab_df_v = ct_pair_gene_pair_metab_df_v[ct_pair_gene_pair_metab_df_v[v].isin(var_subset[v])]

        v_elems = ct_pair_gene_pair_metab_df_v[v].unique().tolist()

        v_scores_df = pd.DataFrame(index=index, columns=v_elems, data=np.zeros((len(index), len(v_elems))))

        for v_elem in v_elems:
            ct_pair_gene_pair_metab_df_v_elem = ct_pair_gene_pair_metab_df_v[
            ct_pair_gene_pair_metab_df_v[v] == v_elem
            ].copy()

            v_scores_df[v_elem] = ct_pair_gene_pair_metab_df_v_elem.groupby("cells")["scores"].sum()

            # Cell type pairs

            if per_ct_pair:
                cell_type_pairs = ct_pair_gene_pair_metab_df_v_elem.cell_type_pair.unique()
                v_ct_pair_scores_df = pd.DataFrame(
                    index=index, columns=cell_type_pairs, data=np.zeros((len(index), len(cell_type_pairs)))
                )
                v_ct_pair_scores_df_tmp = (
                    ct_pair_gene_pair_metab_df_v_elem.groupby(["cells", "cell_type_pair"])["scores"].sum().unstack(level=1)
                )
                for ct_pair in cell_type_pairs:
                    v_ct_pair_scores_df[ct_pair] = v_ct_pair_scores_df_tmp[ct_pair]
                adata.obsm[f"{v_elem}_ct_pair_scores"] = v_ct_pair_scores_df.fillna(0)

            # Cell types

            if per_ct:
                ct_pair_gene_pair_metab_df_v_elem = ct_pair_gene_pair_metab_df_v_elem.explode('cell_type_pair').reset_index(drop=True)
                ct_pair_gene_pair_metab_df_v_elem = ct_pair_gene_pair_metab_df_v_elem.rename(columns={'cell_type_pair': 'cell_type'})
                cell_types = ct_pair_gene_pair_metab_df_v_elem.cell_type.unique()
                v_ct_scores_df = pd.DataFrame(
                    index=index, columns=cell_types, data=np.zeros((len(index), len(cell_types)))
                )
                v_ct_scores_df_tmp = (
                    ct_pair_gene_pair_metab_df_v_elem.groupby(["cells", "cell_type"])["scores"].sum().unstack(level=1)
                )
                for ct in cell_types:
                    v_ct_scores_df[ct] = v_ct_scores_df_tmp[ct]
                adata.obsm[f"{v_elem}_ct_scores"] = v_ct_scores_df.fillna(0)


        adata.obsm[f"{v}_scores"] = v_scores_df.fillna(0)



    # ct_pair_gene_pair_metab_df = ct_pair_gene_pair_metab_df.explode('metabolite').reset_index(drop=True)
    # ct_pair_gene_pair_metab_df = ct_pair_gene_pair_metab_df.explode('cell_type_pair').reset_index(drop=True)

    # use_raw = layer_key == "use_raw"
    # index = adata.raw.obs.index if use_raw else adata.obs_names
    # gene_pair_scores_df = pd.DataFrame(index=index, columns=gene_pairs_dict.keys(), data=np.zeros((len(index), len(gene_pairs_dict.keys()))))
    # metab_scores_df = pd.DataFrame(index=index, columns=metabolites, data=np.zeros((len(index), len(metabolites))))

    # adata.uns["ct_pair_gene_pair_metab_df"] = ct_pair_gene_pair_metab_df

    # # Compute gene pair scores
    # print(f"Computing gene pair scores for {len(gene_pairs_dict.keys())} gene pairs...")



    # for gene_pair in tqdm(gene_pairs_dict.keys()):
    #     ct_pair_gene_pair_metab_df_tmp = ct_pair_gene_pair_metab_df[
    #         ct_pair_gene_pair_metab_df.gene_pair.isin(gene_pairs_dict[gene_pair])
    #     ].copy()

    #     gene_pair_scores_df[gene_pair] = ct_pair_gene_pair_metab_df_tmp.groupby("cells")["scores"].sum()

    #     cell_types = list(np.unique(ct_pair_gene_pair_metab_df_tmp['Cell Type 1'].unique().tolist() + ct_pair_gene_pair_metab_df_tmp['Cell Type 2'].unique().tolist()))
    #     gene_pair_ct_scores_df = pd.DataFrame(
    #         index=index, columns=cell_types, data=np.zeros((len(index), len(cell_types)))
    #     )
    #     gene_pair_ct_scores_df_tmp = (
    #         ct_pair_gene_pair_metab_df_tmp.groupby(["cells", "cell_types"])["scores"].sum().unstack(level=1)
    #     )
    #     for ct in cell_types:
    #         gene_pair_ct_scores_df[ct] = gene_pair_ct_scores_df_tmp[ct]
    #     adata.obsm[f"{gene_pair}_ct_scores"] = gene_pair_ct_scores_df.fillna(0)

    #     cell_type_pairs = ct_pair_gene_pair_metab_df_tmp.cell_type_pair.unique()
    #     gene_pair_ct_pair_scores_df = pd.DataFrame(
    #         index=index, columns=cell_type_pairs, data=np.zeros((len(index), len(cell_type_pairs)))
    #     )
    #     gene_pair_ct_pair_scores_df_tmp = (
    #         ct_pair_gene_pair_metab_df_tmp.groupby(["cells", "cell_type_pair"])["scores"].sum().unstack(level=1)
    #     )
    #     for ct_pair in cell_type_pairs:
    #         gene_pair_ct_pair_scores_df[ct_pair] = gene_pair_ct_pair_scores_df_tmp[ct_pair]
    #     adata.obsm[f"{gene_pair}_ct_pair_scores"] = gene_pair_ct_pair_scores_df.fillna(0)

    # # Compute metabolite scores
    # print(f"Computing metabolite scores for {len(metabolites)} metabolites...")

    # for metabolite in tqdm(metabolites):
    #     ct_pair_gene_pair_metab_df_tmp = ct_pair_gene_pair_metab_df[
    #         ct_pair_gene_pair_metab_df.metabolite == metabolite
    #     ].copy()

    #     metab_scores_df[metabolite] = ct_pair_gene_pair_metab_df_tmp.groupby("cells")["scores"].sum()

    #     cell_types = ct_pair_gene_pair_metab_df_tmp.cell_types.unique()
    #     metab_ct_scores_df = pd.DataFrame(
    #         index=index, columns=cell_types, data=np.zeros((len(index), len(cell_types)))
    #     )
    #     metab_ct_scores_df_tmp = (
    #         ct_pair_gene_pair_metab_df_tmp.groupby(["cells", "cell_types"])["scores"].sum().unstack(level=1)
    #     )
    #     for ct in cell_types:
    #         metab_ct_scores_df[ct] = metab_ct_scores_df_tmp[ct]
    #     adata.obsm[f"{metabolite}_ct_scores"] = metab_ct_scores_df.fillna(0)

    #     cell_type_pairs = ct_pair_gene_pair_metab_df_tmp.cell_type_pair.unique()
    #     metab_ct_pair_scores_df = pd.DataFrame(
    #         index=index, columns=cell_type_pairs, data=np.zeros((len(index), len(cell_type_pairs)))
    #     )
    #     metab_ct_pair_scores_df_tmp = (
    #         ct_pair_gene_pair_metab_df_tmp.groupby(["cells", "cell_type_pair"])["scores"].sum().unstack(level=1)
    #     )
    #     for ct_pair in cell_type_pairs:
    #         metab_ct_pair_scores_df[ct_pair] = metab_ct_pair_scores_df_tmp[ct_pair]
    #     adata.obsm[f"{metabolite}_ct_pair_scores"] = metab_ct_pair_scores_df.fillna(0)

    # adata.obsm["metabolite_scores"] = metab_scores_df.fillna(0)
    # adata.obsm["gene_pair_scores"] = gene_pair_scores_df.fillna(0)


def compute_communication_modules(adata, layer_key, height=0.15):
    metab_scores_df = adata.obsm["metabolite_scores"].T
    gene_pair_scores_df = adata.obsm["gene_pair_scores"].T

    # Metabolites

    print("Computing metabolite modules for {} metabolites...".format(metab_scores_df.shape[0]))

    pairwise_metab_correlations = np.corrcoef(metab_scores_df.rank(axis=1))
    pairwise_metab_correlations[np.arange(metab_scores_df.shape[0]), np.arange(metab_scores_df.shape[0])] = 1.0
    pairwise_metab_correlations = (pairwise_metab_correlations + pairwise_metab_correlations.T) / 2
    assert np.all(pairwise_metab_correlations == pairwise_metab_correlations.T)

    Z_metab = hcluster.complete(squareform(1 - pairwise_metab_correlations))

    metab_modules_map = hcluster.fcluster(Z_metab, height, criterion="distance")
    metab_modules_map = ["Module " + str(mod) for mod in metab_modules_map]

    metab_modules_dict = {}

    for i in range(len(metab_modules_map)):
        metab = metab_scores_df.index[i]
        module = metab_modules_map[i]
        if module not in metab_modules_dict.keys():
            metab_modules_dict[module] = []
        metab_modules_dict[module].append(metab)

    metab_modules = list(metab_modules_dict.keys())

    use_raw = layer_key == "use_raw"
    index = adata.raw.obs.index if use_raw else adata.obs_names
    metab_module_scores = pd.DataFrame(
        index=index, columns=metab_modules, data=np.zeros((len(index), len(metab_modules)))
    )

    metab_module_loadings_dict = {}

    for metab_module in tqdm(metab_modules):
        metabs_mod = metab_modules_dict[metab_module]
        metab_mod_scores_df = metab_scores_df.loc[metabs_mod]

        X = scale(metab_mod_scores_df.T)
        pca = PCA(n_components=1)
        scores = pca.fit_transform(X)
        loadings = pca.components_.T

        sign = pca.components_.mean()  # may need to flip
        if sign < 0:
            scores = scores * -1
            loadings = loadings * -1

        scores = scores[:, 0]
        loadings = loadings[:, 0]

        metab_module_scores[metab_module] = scores
        metab_module_loadings_dict[metab_module] = pd.Series(loadings, index=metabs_mod)

    # metab_module_scores = metab_scores_df.join(pd.DataFrame(metab_modules_map, columns=["metab_module_id"], index = metab_scores_df.index)).groupby("metab_module_id").mean()

    adata.obsm["metabolite_module_scores"] = metab_module_scores
    adata.uns["metabolite_module_loadings"] = metab_module_loadings_dict
    adata.uns["metabolite_modules_dict"] = metab_modules_dict

    # Gene pairs

    print("Computing gene pair modules for {} gene pairs...".format(gene_pair_scores_df.shape[0]))

    pairwise_gene_pair_correlations = np.corrcoef(gene_pair_scores_df.rank(axis=1))
    pairwise_gene_pair_correlations[
        np.arange(gene_pair_scores_df.shape[0]), np.arange(gene_pair_scores_df.shape[0])
    ] = 1.0
    pairwise_gene_pair_correlations = (pairwise_gene_pair_correlations + pairwise_gene_pair_correlations.T) / 2
    assert np.all(pairwise_gene_pair_correlations == pairwise_gene_pair_correlations.T)

    Z_gene_pair = hcluster.complete(squareform(1 - pairwise_gene_pair_correlations))

    gene_pair_modules_map = hcluster.fcluster(Z_gene_pair, height, criterion="distance")
    gene_pair_modules_map = ["Module " + str(mod) for mod in gene_pair_modules_map]

    gene_pair_modules_dict = {}

    for i in range(len(gene_pair_modules_map)):
        gene_pair = gene_pair_scores_df.index[i]
        module = gene_pair_modules_map[i]
        if module not in gene_pair_modules_dict.keys():
            gene_pair_modules_dict[module] = []
        gene_pair_modules_dict[module].append(gene_pair)

    gene_pair_modules = list(gene_pair_modules_dict.keys())

    gene_pair_module_scores = pd.DataFrame(
        index=index, columns=gene_pair_modules, data=np.zeros((len(index), len(gene_pair_modules)))
    )

    gene_pair_module_loadings_dict = {}

    for gene_pair_module in tqdm(gene_pair_modules):
        gene_pairs_mod = gene_pair_modules_dict[gene_pair_module]
        gene_pair_mod_scores_df = gene_pair_scores_df.loc[gene_pairs_mod]

        X = scale(gene_pair_mod_scores_df.T)
        pca = PCA(n_components=1)
        scores = pca.fit_transform(X)
        loadings = pca.components_.T

        sign = pca.components_.mean()  # may need to flip
        if sign < 0:
            scores = scores * -1
            loadings = loadings * -1

        scores = scores[:, 0]
        loadings = loadings[:, 0]

        gene_pair_module_scores[gene_pair_module] = scores
        gene_pair_module_loadings_dict[gene_pair_module] = pd.Series(loadings, index=gene_pairs_mod)

    # gene_pair_module_scores = gene_pair_scores_df.join(pd.DataFrame(gene_pair_modules_map, columns=["gene_pair_module_id"], index = gene_pair_scores_df.index)).groupby("gene_pair_module_id").mean()

    adata.obsm["gene_pair_module_scores"] = gene_pair_module_scores
    adata.uns["gene_pair_module_loadings"] = gene_pair_module_loadings_dict
    adata.uns["gene_pair_modules_dict"] = gene_pair_modules_dict


def compute_interacting_cell_scores(
    adata: Union[str, AnnData],
    cell_communication_df: pd.DataFrame,
    layer_key: Optional[Union[Literal["use_raw"], str]] = None,
    model: Optional[str] = None,
    compute_neighbors_on_key: Optional[str] = None,
    distances_obsp_key: Optional[str] = None,
    deconv_data: Optional[bool] = False,
    cell_type_list: Optional[list] = None,
    cell_type_key: Optional[str] = None,
    barcode_key: Optional[str] = None,
    database_varm_key: Optional[str] = None,
    weighted_graph: Optional[bool] = True,
    neighborhood_radius: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    neighborhood_factor: Optional[int] = 3,
    spot_diameter: Optional[int] = None,
    sample_key: Optional[str] = None,
    func: Optional[Union[Literal["max"], Literal["min"], Literal["mean"], Literal["prod"]]] = 'mean',
):

    # if deconv_data is True:
    #     adata = setup_anndata(
    #         adata,
    #         cell_type_list,
    #         compute_neighbors_on_key,
    #         cell_type_key,
    #         database_varm_key,
    #         sample_key,
    #         )
    #     layer_key=None

    #     if compute_neighbors_on_key is not None:
    #         compute_neighbors(
    #             adata=adata,
    #             compute_neighbors_on_key=compute_neighbors_on_key,
    #             n_neighbors=n_neighbors,
    #             neighborhood_radius=neighborhood_radius,
    #             spot_diameter=spot_diameter,
    #             sample_key=sample_key,
    #             deconv_data=deconv_data,
    #         )
    #     else:
    #         if distances_obsp_key is not None and distances_obsp_key in adata.obsp:
    #             compute_neighbors_from_distances(
    #                 adata=adata,
    #                 distances_obsp_key=distances_obsp_key,
    #                 spot_diameter=spot_diameter,
    #                 sample_key=sample_key,
    #                 deconv_data=deconv_data,
    #             )

    #     if 'weights' not in adata.obsp and 'distances' in adata.obsp:
    #         compute_weights(
    #             adata,
    #             weighted_graph,
    #             neighborhood_factor,
    #         )

    use_raw = layer_key == "use_raw"
    database = adata.varm[database_varm_key] if not use_raw else adata.raw.varm[database_varm_key]

    if barcode_key is not None:
        cells = pd.Series(adata.obs[barcode_key].tolist())
    else:
        cells = adata.obs_names if not use_raw else adata.raw.obs_names

    genes = database.loc[(database != 0).any(axis=1)].index
    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)
    num_umi = counts.sum(axis=0)

    weights = adata.obsp["weights"].tocsr()
    cell_types = adata.obs[cell_type_key] if not use_raw else adata.raw.obs[cell_type_key]

    counts = create_centered_counts(counts, model, num_umi)
    counts = np.nan_to_num(counts)

    cell_communication_df['interacting_cells'] = cell_communication_df.apply(compute_interacting_cell_scores_row, args=(counts, weights, cell_types, cells, genes, func), axis=1)

    return cell_communication_df


def compute_interacting_cell_scores_row(
    row, counts, weights, cell_types, cells, genes, func,
):

    ct_t, ct_u = row['Cell Type 1'], row['Cell Type 2']
    gene_a, gene_b = row['Gene 1'], row['Gene 2']
    gene_a_ind = genes.to_list().index(gene_a)
    gene_b_ind = genes.to_list().index(gene_b)
    cell_type_t_mask = cell_types == ct_t
    cell_type_t_mask = cell_type_t_mask.reset_index(drop=True)
    cell_type_u_mask = cell_types == ct_u
    cell_type_u_mask = cell_type_u_mask.reset_index(drop=True)

    weights_ct = weights[cell_type_t_mask,][:,cell_type_u_mask]
    cells_t = list(cells[cell_type_t_mask])
    cells_u = list(cells[cell_type_u_mask])
    counts_t = counts[:,cell_type_t_mask]
    counts_u = counts[:,cell_type_u_mask]

    interacting_cells = {"cells": [], "scores": []}

    rows, cols = weights_ct.nonzero()
    for i, j in zip(rows, cols):
        w_ij = weights_ct[i, j]

        ai = counts_t[gene_a_ind, i]
        cell_i = cells_t[i]
        bj = counts_u[gene_b_ind, j]
        cell_j = cells_u[j]
        cell_pair = (cell_i, cell_j)

        if not ai == 0 or bj == 0:
            interacting_cells["cells"].append(cell_pair)
            if func == 'mean':
                f_ai_jb = (ai+bj)/2
            elif func == 'min':
                f_ai_jb = ai if ai <= bj else bj
            elif func == 'max':
                f_ai_jb = ai if ai >= bj else bj
            elif func == 'prod':
                f_ai_jb = ai * bj
            interacting_cells["scores"].append(w_ij * f_ai_jb)

    return interacting_cells


def get_metabolic_process(genes, m2h_dict, transporter_df, org='Mouse'):

    genes_list = genes.split('/')
    metab_process = []
    for gene in genes_list:
        gene = gene.strip()
        if org == 'Mouse':
            if gene in m2h_dict.keys():
                gene_h = m2h_dict[gene]
            else:
                continue
        else:
            gene_h = gene

        if gene_h in transporter_df.index:
            gene_h_shape = transporter_df.loc[[gene_h]].shape[0]
            if gene_h_shape > 1:
                for i in range(gene_h_shape):
                    system = transporter_df.loc[gene_h].iloc[i]['System']
            else:
                system = transporter_df.loc[gene_h]['System']
            if system not in metab_process:
                metab_process.append(system)
        else:
            continue

    if len(metab_process) == 0:
        metab_process = 'Unknown'
    else:
        metab_process = ' / '.join(metab_process)

    return metab_process


def get_metabolic_pathway(genes, m2h_dict, transporter_df, org, col):

    genes_list = list(genes)
    metab_pathway = []
    for gene in genes_list:
        gene = gene.strip()
        if org == 'Mouse':
            if gene in m2h_dict.keys():
                gene_h = m2h_dict[gene]
            else:
                continue
        else:
            gene_h = gene

        if gene_h in transporter_df.index:
            gene_h_shape = transporter_df.loc[[gene_h]].shape[0]
            if gene_h_shape > 1:
                for i in range(gene_h_shape):
                    pathway = transporter_df.loc[gene_h].iloc[i][col]
            else:
                pathway = transporter_df.loc[gene_h][col]
            if pathway not in metab_pathway:
                metab_pathway.append(pathway)
        else:
            continue

    if len(metab_pathway) == 0:
        metab_pathway = ('Unknown',)
    else:
        metab_pathway = tuple(metab_pathway)

    return metab_pathway
