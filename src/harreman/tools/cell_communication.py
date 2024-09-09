import itertools
import json
from functools import partial
from random import random
from typing import Literal, Optional, Union
import time
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hcluster
import sparse
import tensorly as tl
from anndata import AnnData
from numba import jit, njit
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from threadpoolctl import threadpool_limits
from scipy.stats.mstats import gmean
from scipy.stats import mannwhitneyu
from ..hotspot import models

from ..preprocessing.anndata import counts_from_anndata
from ..hotspot.local_autocorrelation import compute_local_autocorrelation


def apply_gene_filtering(
    adata: AnnData,
    layer_key: Optional[Union[Literal["use_raw"], str]] = None,
    cell_type_key: Optional[str] = None,
    model: Optional[str] = None,
    autocorrelation_filt: bool = False,
    expression_filt: bool = False,
    de_filt: bool = False,
):
    
    start = time.time()
    print("Applying gene filtering...")
    
    adata.uns['autocorrelation_filt'] = autocorrelation_filt
    adata.uns['expression_filt'] = expression_filt
    adata.uns['de_filt'] = de_filt

    if ('gene_autocorrelation_results' not in adata.uns) and (autocorrelation_filt is True):
        compute_local_autocorrelation(
            adata,
            layer_key,
            adata.uns['database_varm_key'],
            model,
            use_metabolic_genes = False,
        )

    if (expression_filt is True) or (de_filt is True):

        if cell_type_key is None and 'cell_type_key' in adata.uns:
            cell_type_key = adata.uns['cell_type_key']
        elif cell_type_key is None and 'cell_type_key' not in adata.uns:
            raise ValueError('The "cell_type_key" argument needs to be provided.')

        filtered_genes, filtered_genes_ct = filter_genes(adata, layer_key, adata.uns['database_varm_key'], cell_type_key, expression_filt, de_filt, autocorrelation_filt)
        adata.uns['filtered_genes'] = filtered_genes
        adata.uns['filtered_genes_ct'] = filtered_genes_ct
    
    print("Finished applying gene filtering in %.3f seconds" %(time.time()-start))

    return


def filter_genes(adata, layer_key, database_varm_key, cell_type_key, expression_filt, de_filt, autocorrelation_filt):

    if autocorrelation_filt is True:
        autocor_results = adata.uns['gene_autocorrelation_results']

        non_sig_genes = autocor_results.loc[autocor_results.FDR > 0.05].index
        sig_genes = autocor_results.loc[autocor_results.FDR < 0.05].index

        if len(non_sig_genes) == autocor_results.shape[0]:
            raise ValueError(
                "There are no significantly autocorrelated genes."
            )

        counts = counts_from_anndata(adata[:, sig_genes], layer_key, dense=True)

    else:
        use_raw = layer_key == "use_raw"
        database = adata.varm[database_varm_key] if not use_raw else adata.raw.varm[database_varm_key]

        genes = database.loc[(database!=0).any(axis=1)].index
        counts = counts_from_anndata(adata[:, genes], layer_key, dense=True) 


    cell_types = adata.obs[cell_type_key]
    cell_type_list = cell_types.unique().tolist()

    filtered_genes = []
    filtered_genes_ct = {}

    if (expression_filt is True) and (de_filt is True):

        for cell_type in cell_type_list:
            cell_type_mask = cell_types == cell_type
            counts_ct = counts[:,cell_type_mask]
            # genes with high enough expression in each cell type
            gene_expr_bin = np.apply_along_axis(expr_threshold, 1, counts_ct)
            gene_expr_bin_df_ct = pd.DataFrame(gene_expr_bin, index=sig_genes, columns=['gene_expr_bin'])
            gene_expr_bin_df_ct['cell_type'] = cell_type
            gene_expr_bin_df_ct['gene'] = gene_expr_bin_df_ct.index
            # genes overexpressed in each cell type
            no_cell_type_mask = cell_types != cell_type
            counts_no_ct = counts[:,no_cell_type_mask]
            gene_de_results = [de_threshold(counts_ct[i], counts_no_ct[i]) for i in range(counts_ct.shape[0])]
            gene_de_df_ct = pd.DataFrame(index=sig_genes)
            gene_de_df_ct[['stat', 'pval', 'cohens_d']] = pd.Series(gene_de_results).tolist()
            gene_de_df_ct['cell_type'] = cell_type
            gene_de_df_ct['gene'] = gene_de_df_ct.index
            if cell_type == cell_type_list[0]:
                gene_expr_bin_df = gene_expr_bin_df_ct
                gene_de_df = gene_de_df_ct
            else:
                gene_expr_bin_df = pd.concat([gene_expr_bin_df, gene_expr_bin_df_ct])
                gene_de_df = pd.concat([gene_de_df, gene_de_df_ct])

        gene_de_df['FDR'] = multipletests(gene_de_df["pval"], method="fdr_bh")[1]

        for cell_type in cell_type_list:
            gene_de_df_ct = gene_de_df[gene_de_df['cell_type'] == cell_type]
            gene_expr_bin_df_ct = gene_expr_bin_df[gene_expr_bin_df['cell_type'] == cell_type]
            gene_de_ct_sig = gene_de_df_ct[(gene_de_df_ct['FDR'] < 0.05) & gene_de_df_ct['cohens_d'] > 0]['gene'].tolist()
            gene_expr_bin_ct = gene_expr_bin_df_ct[gene_expr_bin_df_ct['gene_expr_bin'] == True]['gene'].tolist()

            filtered_genes_ct_list = list(set(gene_de_ct_sig) & set(gene_expr_bin_ct))

            filtered_genes_ct[cell_type] = filtered_genes_ct_list
            filtered_genes.append(filtered_genes_ct_list)

        filtered_genes = [gene for ct_genes in filtered_genes for gene in ct_genes]
        filtered_genes = list(np.unique(filtered_genes))


    elif (expression_filt is True) and (de_filt is False):

        for cell_type in cell_type_list:
            cell_type_mask = cell_types == cell_type
            counts_ct = counts[:,cell_type_mask]
            # genes with high enough expression in each cell type
            gene_expr_bin = np.apply_along_axis(expr_threshold, 1, counts_ct)
            gene_expr_bin_df_ct = pd.DataFrame(gene_expr_bin, index=sig_genes, columns=['gene_expr_bin'])
            gene_expr_bin_ct = gene_expr_bin_df_ct[gene_expr_bin_df_ct['gene_expr_bin'] is True].index.tolist()

            filtered_genes_ct[cell_type] = gene_expr_bin_ct
            filtered_genes.append(gene_expr_bin_ct)

        filtered_genes = [gene for ct_genes in filtered_genes for gene in ct_genes]
        filtered_genes = list(np.unique(filtered_genes))


    elif (expression_filt is False) and (de_filt is True):

        for cell_type in cell_type_list:
            cell_type_mask = cell_types == cell_type
            counts_ct = counts[:,cell_type_mask]
            # genes overexpressed in each cell type
            no_cell_type_mask = cell_types != cell_type
            counts_no_ct = counts[:,no_cell_type_mask]
            gene_de_results = [de_threshold(counts_ct[i], counts_no_ct[i]) for i in range(counts_ct.shape[0])]
            gene_de_df_ct = pd.DataFrame(index=sig_genes)
            gene_de_df_ct[['stat', 'pval', 'cohens_d']] = pd.Series(gene_de_results).tolist()
            gene_de_df_ct['cell_type'] = cell_type
            gene_de_df_ct['gene'] = gene_de_df_ct.index
            if cell_type == cell_type_list[0]:
                gene_de_df = gene_de_df_ct
            else:
                gene_de_df = pd.concat([gene_de_df, gene_de_df_ct])

        gene_de_df['FDR'] = multipletests(gene_de_df["pval"], method="fdr_bh")[1]

        for cell_type in cell_type_list:
            gene_de_df_ct = gene_de_df[gene_de_df['cell_type'] == cell_type]
            gene_de_ct_sig = gene_de_df_ct[(gene_de_df_ct['FDR'] < 0.05) & gene_de_df_ct['cohens_d'] > 0]['gene'].tolist()

            filtered_genes_ct[cell_type] = gene_de_ct_sig
            filtered_genes.append(gene_de_ct_sig)

        filtered_genes = [gene for ct_genes in filtered_genes for gene in ct_genes]
        filtered_genes = list(np.unique(filtered_genes))


    return filtered_genes, filtered_genes_ct


def expr_threshold(row):
    return (row > 0).sum()/len(row) > 0.2


def de_threshold(row1, row2):
    stat, pval = mannwhitneyu(row1, row2, alternative='greater')
    c_d = cohens_d(row1, row2)
    return (stat, pval, c_d)


def cohens_d(x, y):
    pooled_std = np.sqrt(((len(x)-1) * np.var(x, ddof=1)
                          + (len(y)-1) * np.var(y, ddof=1)) /
                             (len(x) + len(y) - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std


def compute_gene_pairs(
    adata: AnnData,
    layer_key: Optional[Union[Literal["use_raw"], str]] = None,
    cell_type_key: Optional[str] = None,
    cell_type_pairs: Optional[list] = None,
    ct_specific: Optional[bool] = True,
):
    
    start = time.time()
    print("Computing gene pairs...")

    from_value_to_type = {
        'LR': { -1.0: "REC", 1.0: "LIG"}, 
        'transporter': {-1.0: "IMP", 1.0: "EXP", 2.0: "IMP-EXP"}
    }

    if 'layer_key' in adata.uns and layer_key is None:
        layer_key = adata.uns['layer_key']

    if 'autocorrelation_filt' in adata.uns:
        autocorrelation_filt = adata.uns['autocorrelation_filt']
        expression_filt = adata.uns['expression_filt']
        de_filt = adata.uns['de_filt']
    else:
        autocorrelation_filt = False
        expression_filt = False
        de_filt = False
    
    if 'cell_type_key' in adata.uns and cell_type_key is None:
        cell_type_key = adata.uns['cell_type_key']
    elif 'cell_type_key' not in adata.uns and cell_type_key is None and ct_specific:
        raise ValueError('Please provide the "cell_type_key" argument.')

    use_raw = layer_key == "use_raw"
    genes = adata.raw.var.index if use_raw else adata.var_names

    if ct_specific:
        cell_types = adata.obs[cell_type_key] if not use_raw else adata.raw.obs[cell_type_key]
        cell_types = cell_types.values.astype(str)

    database = adata.varm["database"]

    if (expression_filt is True) or (de_filt is True):
        filtered_genes = adata.uns["filtered_genes"]
        filtered_genes_ct = adata.uns["filtered_genes_ct"]
    elif autocorrelation_filt is True:
        autocor_results = adata.uns["gene_autocorrelation_results"]
        filtered_genes = autocor_results[autocor_results.FDR < 0.05].index.tolist()
    else:
        filtered_genes = genes

    if "filtered_genes_ct" not in adata.uns and ct_specific:
        filtered_genes_ct = {}
        for ct in cell_types:
            filtered_genes_ct[ct] = filtered_genes

    if len(filtered_genes) == 0:
        raise ValueError("No genes have passed the filters.")

    non_sig_genes = [g for g in genes if g not in filtered_genes]

    database.loc[non_sig_genes] = 0
    cols_keep = [col for col in database.columns if (np.unique(database[col]) != 0).sum() > 1 or database[col][database[col] != 0].unique().tolist() == [2]]
    database = database[cols_keep].copy()
    adata.varm["database"] = database

    gene_pairs_per_metabolite = {}
    gene_pairs = []
    ct_pairs = []

    weights = adata.obsp["weights"]

    if cell_type_pairs is None and ct_specific:
        cell_type_list = list(filtered_genes_ct.keys())
        cell_type_pairs = list(itertools.combinations(cell_type_list, 2))

    if ct_specific:
        cell_type_pairs_df = pd.Series(cell_type_pairs)
        cell_type_pairs_int = cell_type_pairs_df.apply(get_interacting_cell_type_pairs, args=(weights, cell_types))
        cell_type_pairs = cell_type_pairs_df[cell_type_pairs_int].tolist()
        gene_pairs_per_ct_pair = {}

    if adata.uns['database']  == 'both':
        metabolites = adata.uns['metabolite_database'].Metabolite.tolist()
        LR_pairs = adata.uns['LR_database'].index.tolist()

    for metabolite in database.columns:
        metab_genes = database[database[metabolite] != 0].index.tolist()
        if len(metab_genes) == 0:
            continue
        gene_pairs_per_metabolite[metabolite] = {"gene_pair": [], "gene_type": []}

        if adata.uns['database'] == 'both':
            int_type = 'transporter' if metabolite in metabolites else 'LR' if metabolite in LR_pairs else None
            if int_type is None:
                raise ValueError('The "metabolite" variable needs to be either a metabolite or a LR pair.')
        else:
            int_type = adata.uns['database']

        if int_type == 'transporter':
            all_pairs = list(set(itertools.combinations_with_replacement(metab_genes, 2)) | set(itertools.permutations(metab_genes, 2)))
        else:
            ligand = adata.uns['ligand'].loc[metabolite].dropna().tolist()
            ligand = ligand[0] if len(ligand) == 1 else ligand
            receptor = adata.uns['receptor'].loc[metabolite].dropna().tolist()
            receptor = receptor[0] if len(receptor) == 1 else receptor
            if len(ligand) == 0 or len(receptor) == 0:
                continue
            all_pairs = [(ligand, receptor)]
        
        for pair in all_pairs:
            var1, var2 = pair
            var1_value = database.loc[var1, metabolite] if type(var1) is str else [database.loc[var1_, metabolite] for var1_ in var1]
            var1_value = list(np.unique(var1_value)) if type(var1_value) is list else var1_value
            var1_value = [val for val in var1_value if val!=0] if type(var1_value) is list and len(var1_value)>1 else var1_value
            var1_value = var1_value[0] if type(var1_value) is list and len(var1_value) == 1 else var1_value
            var2_value = database.loc[var2, metabolite] if type(var2) is str else [database.loc[var2_, metabolite] for var2_ in var2]
            var2_value = list(np.unique(var2_value)) if type(var2_value) is list else var2_value
            var2_value = [val for val in var2_value if val!=0] if type(var2_value) is list and len(var2_value)>1 else var2_value
            var2_value = var2_value[0] if type(var2_value) is list and len(var2_value) == 1 else var2_value
            if not var1_value or not var2_value:
                continue
            if not (var1_value == 1.0 and var2_value == 1.0) or (var1_value == -1.0 and var2_value == -1.0):
                var1_type = from_value_to_type[int_type][var1_value]
                var2_type = from_value_to_type[int_type][var2_value]
                gene_pairs_per_metabolite[metabolite]["gene_pair"].append((var1, var2))
                gene_pairs_per_metabolite[metabolite]["gene_type"].append((var1_type, var2_type))
                if (var1, var2) not in gene_pairs: # ((var1, var2) not in gene_pairs) and ((var2, var1) not in gene_pairs)
                    gene_pairs.append(pair)
                    if ct_specific:
                        for ct_pair in cell_type_pairs:
                            ct_1, ct_2 = ct_pair
                            ct_pair_str = (ct_1, ct_2)
                            var1_in_ct1 = var1 in filtered_genes_ct[ct_1] if type(var1) is str else np.any([var in filtered_genes_ct[ct_1] for var in var1])
                            var2_in_ct2 = var2 in filtered_genes_ct[ct_2] if type(var2) is str else np.any([var in filtered_genes_ct[ct_2] for var in var2])
                            if var1_in_ct1 and var2_in_ct2:
                                if ct_pair_str not in gene_pairs_per_ct_pair.keys():
                                    gene_pairs_per_ct_pair[ct_pair_str] = []
                                gene_pairs_per_ct_pair[ct_pair_str].append(pair)
                                if ct_pair not in ct_pairs:
                                    ct_pairs.append(ct_pair)

    if "gene_pairs" not in adata.uns:
        adata.uns["gene_pairs"] = gene_pairs
    if "cell_type_pairs" not in adata.uns and ct_specific:
        adata.uns["cell_type_pairs"] = ct_pairs
    if "gene_pairs_per_metabolite" not in adata.uns:
        adata.uns["gene_pairs_per_metabolite"] = gene_pairs_per_metabolite
    if "gene_pairs_per_ct_pair" not in adata.uns and ct_specific:
        adata.uns["gene_pairs_per_ct_pair"] = gene_pairs_per_ct_pair
    
    print("Finished computing gene pairs in %.3f seconds" %(time.time()-start))

    return


def compute_cell_communication(
    adata: AnnData,
    layer_key_p_test: Optional[Union[Literal["use_raw"], str]] = None,
    layer_key_np_test: Optional[Union[Literal["use_raw"], str]] = None,
    model: Optional[str] = None,
    cell_type_key: Optional[str] = None,
    center_counts_for_np_test: Optional[bool] = False,
    M: Optional[int] = 1000,
    test: Optional[Union[Literal["parametric"], Literal["non-parametric"], Literal["both"]]] = None,
    # njobs: Optional[int] = 1,
):
    
    start = time.time()
    print("Starting cell-cell communication analysis...")

    adata.uns['ccc_results'] = {}

    if test not in ['both', 'parametric', 'non-parametric']:
        raise ValueError('The "test" variable should be one of ["both", "parametric", "non-parametric"].')
    
    if 'cell_type_key' in adata.uns and cell_type_key is None:
        cell_type_key = adata.uns['cell_type_key']
    # elif 'cell_type_key' not in adata.uns and cell_type_key is None:
    #     raise ValueError('Please provide the "cell_type_key" argument.')
    
    adata.uns['layer_key_p_test'] = layer_key_p_test
    adata.uns['layer_key_np_test'] = layer_key_np_test
    adata.uns['model'] = model
    adata.uns['cell_type_key'] = cell_type_key
    adata.uns['center_counts_for_np_test'] = center_counts_for_np_test
    adata.uns['model'] = model

    # with threadpool_limits(limits=njobs, user_api='blas'):
    run_cell_communication_analysis(adata, layer_key_p_test, layer_key_np_test, model, cell_type_key, center_counts_for_np_test, M, test)

    print("Obtaining the communication results...")
    if cell_type_key:
        get_cell_communication_results(
            adata,
            adata.uns["genes"],
            layer_key_p_test,
            layer_key_np_test,
            model,
            adata.uns["cell_types"],
            adata.uns["cell_type_pairs"],
            adata.uns["D"],
            test,
        )
    else:
        get_cell_communication_results_no_ct(
            adata,
            adata.uns["genes"],
            layer_key_p_test,
            layer_key_np_test,
            model,
            adata.uns["D"],
            test,
        )


    print("Finished computing cell-cell communication analysis in %.3f seconds" %(time.time()-start))

    return


def run_cell_communication_analysis(
    adata,
    layer_key_p_test,
    layer_key_np_test,
    model,
    cell_type_key,
    center_counts_for_np_test,
    M,
    test,
):
    
    use_raw = (layer_key_p_test == "use_raw") & (layer_key_np_test == "use_raw")

    cells = adata.obs_names if not use_raw else adata.raw.obs.index
    cells = cells.values.astype(str)

    weights = adata.obsp["weights"]
    if cell_type_key:
        cell_types = adata.obs[cell_type_key] if not use_raw else adata.raw.obs[cell_type_key]
    gene_pairs = adata.uns["gene_pairs"]
    genes = list(flatten(gene_pairs))
    adata.uns["genes"] = genes
    cell_type_pairs = adata.uns["cell_type_pairs"] if "cell_type_pairs" in adata.uns.keys() else None
    gene_pairs_per_ct_pair = adata.uns["gene_pairs_per_ct_pair"] if "gene_pairs_per_ct_pair" in adata.uns.keys() else None

    gene_pairs_ind = []
    for pair in gene_pairs:
        var1, var2 = pair
        var1_ind = genes.index(var1) if type(var1) is not list else [genes.index(var) for var in var1 if var in genes]
        var2_ind = genes.index(var2) if type(var2) is not list else [genes.index(var) for var in var2 if var in genes]
        pair_tuple = (var1_ind, var2_ind)
        gene_pairs_ind.append(pair_tuple)
    
    adata.uns["gene_pairs_ind"] = gene_pairs_ind

    if gene_pairs_per_ct_pair:
        gene_pairs_per_ct_pair_ind = {}
        for ct_pair in gene_pairs_per_ct_pair.keys():
            gene_pairs = gene_pairs_per_ct_pair[ct_pair]
            gene_pairs_per_ct_pair_ind[ct_pair] = []
            for pair in gene_pairs:
                var1, var2 = pair
                var1_ind = genes.index(var1) if type(var1) is not list else [genes.index(var) for var in var1 if var in genes]
                var2_ind = genes.index(var2) if type(var2) is not list else [genes.index(var) for var in var2 if var in genes]
                pair_tuple = (var1_ind, var2_ind)
                gene_pairs_per_ct_pair_ind[ct_pair].append(pair_tuple)
    
        adata.uns["gene_pairs_per_ct_pair_ind"] = gene_pairs_per_ct_pair_ind
    else:
        gene_pairs_per_ct_pair_ind = None

    if cell_type_key:
        weights = sparse.COO.from_scipy_sparse(weights)
        weights_ct_pairs = get_ct_pair_weights(
            weights, cell_type_pairs, cell_types
        )
    
    row_degrees = weights_ct_pairs.sum(axis=2).todense() if cell_type_key else weights.sum(axis=1).A1
    col_degrees = weights_ct_pairs.sum(axis=1).todense() if cell_type_key else weights.sum(axis=0).A1
    D = row_degrees + col_degrees
    adata.uns["D"] = D

    if test in ['parametric', 'both']:
        counts_1 = np.vstack([counts_from_anndata(adata[:, gene_pair[0]], layer_key_p_test, dense=True) if type(gene_pair[0]) is not list else gmean(counts_from_anndata(adata[:, gene_pair[0]], layer_key_p_test, dense=True), axis=0) for gene_pair in gene_pairs])
        counts_2 = np.vstack([counts_from_anndata(adata[:, gene_pair[1]], layer_key_p_test, dense=True) if type(gene_pair[1]) is not list else gmean(counts_from_anndata(adata[:, gene_pair[1]], layer_key_p_test, dense=True), axis=0) for gene_pair in gene_pairs])

        num_umi_1 = counts_1.sum(axis=0)
        counts_1_std = create_centered_counts(counts_1, model, num_umi_1)
        counts_1_std = np.nan_to_num(counts_1_std)

        num_umi_2 = counts_2.sum(axis=0)
        counts_2_std = create_centered_counts(counts_2, model, num_umi_2)
        counts_2_std = np.nan_to_num(counts_2_std)

        adata.uns['ccc_results']['p']={}
        adata.uns['ccc_results']['p']['cs'] = compute_CCC_scores(counts_1_std.T, counts_2_std.T, weights_ct_pairs) if cell_type_key else compute_CCC_scores(counts_1_std.T, counts_2_std.T, weights)

        if cell_type_key:
            weights_ct_pairs_sq_data = weights_ct_pairs.data ** 2
            weights_ct_pairs_sq = sparse.COO(weights_ct_pairs.coords, weights_ct_pairs_sq_data, shape=weights_ct_pairs.shape)
            Wtot2 = sparse.sum(weights_ct_pairs_sq, axis=(1, 2))
        else:
            Wtot2 = (weights ** 2).sum()

        counts = counts_from_anndata(adata[:, genes], layer_key_p_test, dense=True)
        num_umi = counts.sum(axis=0)
        counts = create_centered_counts(counts, model, num_umi)
        counts = np.nan_to_num(counts)
        counts = sparse.COO.from_numpy(counts) if cell_type_key else counts

        eg2s = conditional_eg2_cellcom(counts, weights_ct_pairs) if cell_type_key else conditional_eg2_cellcom(counts, weights)

        if cell_type_key:
            compute_p_results_partial = partial(
                compute_p_results,
                cell_type_pairs=cell_type_pairs,
                gene_pair_cor=adata.uns['ccc_results']['p']['cs'],
                gene_pairs_ind=gene_pairs_ind,
                gene_pairs_per_ct_pair_ind=gene_pairs_per_ct_pair_ind,
                Wtot2=Wtot2,
                eg2s=eg2s,
            )

            p_results = list(map(compute_p_results_partial, cell_type_pairs))
            C_scores = np.vstack([x[0] for x in p_results])
            Z_scores = np.vstack([x[1] for x in p_results])
        else:
            p_results = compute_p_results_no_ct(adata.uns['ccc_results']['p']['cs'], gene_pairs_ind, Wtot2, eg2s)
            C_scores = p_results[0]
            Z_scores = p_results[1]

            genes = pd.Series([gene for gene_pair in gene_pairs for gene in gene_pair]).drop_duplicates().tolist()
            genes = [tuple(i) if isinstance(i, list) else i for i in genes]
            gene_pairs_ = [(tuple(gp_1) if isinstance(gp_1, list) else gp_1, tuple(gp_2) if isinstance(gp_2, list) else gp_2) for gp_1, gp_2 in gene_pairs]

            lc_zs = pd.DataFrame(np.zeros((len(genes), len(genes))), index=genes, columns=genes)
            for i, gene_pair in enumerate(gene_pairs_):
                gp_1, gp_2 = gene_pair
                gp_1_ind = genes.index(gp_1)
                gp_2_ind = genes.index(gp_2)
                lc_zs.iloc[gp_1_ind, gp_2_ind] = Z_scores[i]
            
            np.fill_diagonal(lc_zs.values, 0)
            lc_zs = lc_zs/2 + lc_zs.T/2
            adata.uns['lc_zs'] = lc_zs

        Z_pvals = norm.sf(Z_scores)

        adata.uns['ccc_results']['p']['stat'] = eg2s
        adata.uns['ccc_results']['p']['cs'] = C_scores
        adata.uns['ccc_results']['p']['Z'] = Z_scores
        adata.uns['ccc_results']['p']['Z_pval'] = Z_pvals
        adata.uns['ccc_results']['p']['Z_FDR'] = multipletests(Z_pvals.flatten(), method="fdr_bh")[1].reshape(Z_pvals.shape)

    if test in ["non-parametric", "both"]:

        counts_1 = np.vstack([counts_from_anndata(adata[:, gene_pair[0]], layer_key_np_test, dense=True) if type(gene_pair[0]) is not list else gmean(counts_from_anndata(adata[:, gene_pair[0]], layer_key_np_test, dense=True), axis=0) for gene_pair in gene_pairs])
        counts_2 = np.vstack([counts_from_anndata(adata[:, gene_pair[1]], layer_key_np_test, dense=True) if type(gene_pair[1]) is not list else gmean(counts_from_anndata(adata[:, gene_pair[1]], layer_key_np_test, dense=True), axis=0) for gene_pair in gene_pairs])

        if center_counts_for_np_test:
            num_umi_1 = counts_1.sum(axis=0)
            counts_1 = create_centered_counts(counts_1, model, num_umi_1)
            counts_1 = np.nan_to_num(counts_1)

            num_umi_2 = counts_2.sum(axis=0)
            counts_2 = create_centered_counts(counts_2, model, num_umi_2)
            counts_2 = np.nan_to_num(counts_2)

        adata.uns['ccc_results']['np']={}
        adata.uns['ccc_results']['np']['perm_cs'] = np.zeros((len(cell_type_pairs), counts_1.shape[0], M)).astype(np.float16) if cell_type_key else np.zeros((counts_1.shape[0], M)).astype(np.float16)
        if center_counts_for_np_test and test == 'both':
            adata.uns['ccc_results']['np']['cs'] = adata.uns['ccc_results']['p']['cs']
        else:
            adata.uns['ccc_results']['np']['cs'] = compute_CCC_scores(counts_1.T, counts_2.T, weights_ct_pairs) if cell_type_key else compute_CCC_scores(counts_1.T, counts_2.T, weights)
        
        for i in tqdm(range(M)):
            idx = np.random.permutation(counts_1.shape[1])
            counts_1, counts_2 = counts_1[:, idx], counts_2[:, idx]

            if cell_type_key:
                cell_types_perm = pd.Series(cell_types[idx])
                weigths_ct_pairs_perm = get_ct_pair_weights(
                    weights, cell_type_pairs, cell_types_perm
                )
                adata.uns['ccc_results']['np']['perm_cs'][:, :, i] = compute_CCC_scores(counts_1.T, counts_2.T, weigths_ct_pairs_perm)
            else:
                adata.uns['ccc_results']['np']['perm_cs'][:, i] = compute_CCC_scores(counts_1.T, counts_2.T, weights)
        
        if cell_type_key:
            x = np.sum(adata.uns['ccc_results']['np']['perm_cs'] > adata.uns['ccc_results']['np']['cs'][:, :, np.newaxis], axis=2)
        else:
            x = np.sum(adata.uns['ccc_results']['np']['perm_cs'] > adata.uns['ccc_results']['np']['cs'][:, np.newaxis], axis=1)
        pvals = (x + 1) / (M + 1)

        if cell_type_key:
            compute_np_results_partial = partial(
                compute_np_results,
                cell_type_pairs=cell_type_pairs,
                gene_pair_cor=adata.uns['ccc_results']['np']['cs'],
                pvals=pvals,
                gene_pairs_ind=gene_pairs_ind,
                gene_pairs_per_ct_pair_ind=gene_pairs_per_ct_pair_ind,
            )
            np_results = list(map(compute_np_results_partial, cell_type_pairs))
            C_scores = np.vstack([x[0] for x in np_results])
            pvals = np.vstack([x[1] for x in np_results])
        else:
            np_results = compute_np_results_no_ct(adata.uns['ccc_results']['np']['cs'], pvals, gene_pairs_ind)
            C_scores = np_results[0]
            pvals = np_results[1]
        
        adata.uns['ccc_results']['np']['cs'] = C_scores
        adata.uns['ccc_results']['np']['pval'] = pvals
        adata.uns['ccc_results']['np']['FDR'] = multipletests(pvals.flatten(), method="fdr_bh")[1].reshape(pvals.shape) if cell_type_key else multipletests(pvals, method="fdr_bh")[1]

    adata.uns["cell_types"] = cell_types.tolist() if cell_type_key else None
    
    return


def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


def select_significant_interactions(
    adata: AnnData,
    test: Optional[Union[Literal["parametric"], Literal["non-parametric"]]] = "parametric",
    threshold: Optional[float] = 0.05,
):

    if test == 'parametric':
        FDR_values = adata.uns['ccc_results']['cell_com_df']['Z_FDR'].values
        C_values = adata.uns['ccc_results']['cell_com_df']['C_p'].values
    elif test == 'non-parametric':
        FDR_values = adata.uns['ccc_results']['cell_com_df']['FDR_np'].values
        C_values = adata.uns['ccc_results']['cell_com_df']['C_np'].values
    else:
        raise ValueError('The "test" variable should be one of ["parametric", "non-parametric"].')

    adata.uns['ccc_results']['cell_com_df']['selected'] = (FDR_values < threshold) & (C_values > 0) if test == 'non-parametric' else (FDR_values < threshold)
    cell_com_df = adata.uns['ccc_results']['cell_com_df']
    adata.uns['ccc_results']['cell_com_df_sig'] = cell_com_df[cell_com_df.selected == True].copy()

    return


def compute_metabolite_scores(
    adata: Union[str, AnnData],
    var: str,
    func: Optional[Union[Literal["sum"], Literal["mean"], Literal["max"]]] = 'mean',
):
    
    cell_type_key = adata.uns['cell_type_key']

    cell_communication_df = adata.uns['ccc_results']['cell_com_df_sig'].copy()
    cell_communication_df[['Gene 1', 'Gene 2']] = cell_communication_df[['Gene 1', 'Gene 2']].map(lambda x: tuple(x) if isinstance(x, list) else x)
    # cell_communication_df['gene_pair'] = cell_communication_df.apply(lambda row: ensure_tuple((row['Gene 1'], row['Gene 2'])), axis=1)
    cell_communication_df['gene_pair'] = cell_communication_df.apply(lambda row: (row['Gene 1'], row['Gene 2']), axis=1)

    gene_pairs_per_metabolite = adata.uns['gene_pairs_per_metabolite']

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

    adata.uns['metabolite_gene_pair_df'] = metabolite_gene_pair_df

    metabolite_gene_pair_df['gene_pair'] = metabolite_gene_pair_df.apply(lambda row: ensure_tuple(row['gene_pair']), axis=1)

    cell_communication_df = pd.merge(cell_communication_df, metabolite_gene_pair_df, on='gene_pair', how='inner')

    adata.uns['ccc_results']['cell_com_df_sig_metab'] = cell_communication_df
    if func == 'sum':
        cell_communication_df = cell_communication_df.groupby(['Cell Type 1', 'Cell Type 2', 'metabolite'])[var].sum().reset_index() if cell_type_key else cell_communication_df.groupby(['metabolite'])[var].sum().reset_index()
    elif func == 'mean':
        cell_communication_df = cell_communication_df.groupby(['Cell Type 1', 'Cell Type 2', 'metabolite'])[var].mean().reset_index() if cell_type_key else cell_communication_df.groupby(['metabolite'])[var].sum().reset_index()
    elif func == 'max':
        cell_communication_df = cell_communication_df.groupby(['Cell Type 1', 'Cell Type 2', 'metabolite'])[var].max().reset_index() if cell_type_key else cell_communication_df.groupby(['metabolite'])[var].sum().reset_index()
    else:
        raise ValueError('The "func" variable should be one of ["sum", "mean", "max"].')

    adata.uns['ccc_results'][f'cell_com_df_sig_metab_scores_{var}_{func}'] = cell_communication_df

    return


def compute_interacting_cell_scores(
    adata: Union[str, AnnData],
    func: Optional[Union[Literal["max"], Literal["min"], Literal["mean"], Literal["prod"]]] = 'mean',
    center_counts: Optional[bool] = False,
    test: Optional[Union[Literal["parametric"], Literal["non-parametric"]]] = None,
):
    
    # To visualize the results, the user should ask for a given ct pair and a metabolite/LR pair. More than one plot can also be visualized. Output an error if the asked ct pair/metabolite is not significant.

    model = adata.uns["model"]
    if not test:
        raise ValueError('The "test" variable should be one of ["parametric", "non-parametric"].')
    layer_key = adata.uns["layer_key_p_test"] if test == 'parametric' else adata.uns["layer_key_np_test"]
    use_raw = layer_key == "use_raw"

    cell_communication_df = adata.uns['ccc_results']['cell_com_df_sig'].copy()
    cell_communication_df[['Gene 1', 'Gene 2']] = cell_communication_df[['Gene 1', 'Gene 2']].map(lambda x: tuple(x) if isinstance(x, list) else x)

    gene_pairs_per_ct_pair = adata.uns['gene_pairs_per_ct_pair'] if 'gene_pairs_per_ct_pair' in adata.uns else None
    gene_pairs_per_metabolite = adata.uns['gene_pairs_per_metabolite']

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

    if gene_pairs_per_ct_pair:
        gene_pairs_per_ct_pair_sig = {}
        for ct_pair in gene_pairs_per_ct_pair.keys():
            ct_1, ct_2 = ct_pair
            cell_com_df_ct_pair = cell_communication_df[(cell_communication_df['Cell Type 1'] == ct_1) & (cell_communication_df['Cell Type 2'] == ct_2)]
            if cell_com_df_ct_pair.shape[0] == 0:
                continue
            gene_pairs_per_ct_pair_sig[ct_pair] = []
            for gene_pair in gene_pairs_per_ct_pair[ct_pair]:
                gene_1, gene_2 = gene_pair
                gene_1 = gene_1 if type(gene_1) is str else tuple(gene_1)
                gene_2 = gene_2 if type(gene_2) is str else tuple(gene_2)
                cell_com_df_ct_pair_gene_pair = cell_com_df_ct_pair[(cell_com_df_ct_pair['Gene 1'] == gene_1) & (cell_com_df_ct_pair['Gene 2'] == gene_2)]
                if cell_com_df_ct_pair_gene_pair.shape[0] == 0:
                    continue
                gene_pairs_per_ct_pair_sig[ct_pair].append(gene_pair)

    if 'barcode_key' in adata.uns:
        barcode_key = adata.uns['barcode_key']
        cells = pd.Series(adata.obs[barcode_key].tolist())
    else:
        cells = adata.obs_names if not use_raw else adata.raw.obs_names

    genes = adata.uns['genes']
    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)
    num_umi = counts.sum(axis=0)
    if center_counts:
        counts = create_centered_counts(counts, model, num_umi)
        counts = np.nan_to_num(counts)

    weights = adata.obsp["weights"].tocsr()
    cell_type_key = adata.uns['cell_type_key']
    if cell_type_key:
        cell_types = adata.obs[cell_type_key] if not use_raw else adata.raw.obs[cell_type_key]

        extract_ct_pair_interacting_scores_partial = partial(
            extract_ct_pair_interacting_scores,
            gene_pairs_per_ct_pair_sig=gene_pairs_per_ct_pair_sig,
            metabolite_gene_pair_df=metabolite_gene_pair_df,
            counts=counts,
            weights=weights,
            cell_types=cell_types,
            cells=cells,
            genes=genes,
            func=func,
        )
        results = list(map(extract_ct_pair_interacting_scores_partial, gene_pairs_per_ct_pair_sig))
        interacting_cell_scores = {k: v for d in results for k, v in d.items()}
    
    else:
        gene_pairs = adata.uns['gene_pairs']

        gene_pairs_sig = []
        for gene_pair in gene_pairs:
            gene_1, gene_2 = gene_pair
            gene_1 = gene_1 if type(gene_1) is str else tuple(gene_1)
            gene_2 = gene_2 if type(gene_2) is str else tuple(gene_2)
            cell_com_df_gene_pair = cell_communication_df[(cell_communication_df['Gene 1'] == gene_1) & (cell_communication_df['Gene 2'] == gene_2)]
            if cell_com_df_gene_pair.shape[0] == 0:
                continue
            gene_pairs_sig.append(gene_pair)
        
        compute_interacting_cell_scores_no_ct_row_partial = partial(
            compute_interacting_cell_scores_no_ct_row,
            counts=counts,
            weights=weights,
            cells=cells,
            genes=genes,
            func=func,
        )
        results = pd.concat(list(map(compute_interacting_cell_scores_no_ct_row_partial, gene_pairs_sig)), axis=1)

        metabolite_gene_pair_df = metabolite_gene_pair_df[metabolite_gene_pair_df.gene_pair.isin(results.columns)]

        results_transposed = results.T
        results_transposed.index = ['-'.join('-'.join(elem) if isinstance(elem, list) else elem for elem in tup) for tup in results_transposed.index]
        df_metabolites_grouped = pd.DataFrame(index=results.index)

        for metabolite, group in metabolite_gene_pair_df.groupby('metabolite'):
            gene_pairs = group['gene_pair'].tolist()
            gene_pairs = ['-'.join('-'.join(elem) if isinstance(elem, list) else elem for elem in tup) for tup in gene_pairs]
            df_metabolites_grouped[metabolite] = results_transposed.loc[gene_pairs].sum()

        unique_ind = len(df_metabolites_grouped.index.unique()) == df_metabolites_grouped.shape[0]
        interacting_cell_scores = df_metabolites_grouped.groupby(df_metabolites_grouped.index).sum() if not unique_ind else df_metabolites_grouped
        

    adata.uns['interacting_cell_scores'] = interacting_cell_scores

    return


def ensure_tuple(x):
    return tuple(tuple(i) if isinstance(i, list) else i for i in x)


def extract_ct_pair_interacting_scores(
    ct_pair, gene_pairs_per_ct_pair_sig, metabolite_gene_pair_df, counts, weights, cell_types, cells, genes, func
):
    
    gene_pairs = gene_pairs_per_ct_pair_sig[ct_pair]

    compute_interacting_cell_scores_row_partial = partial(
        compute_interacting_cell_scores_row,
        ct_pair=ct_pair,
        counts=counts,
        weights=weights,
        cell_types=cell_types,
        cells=cells,
        genes=genes,
        func=func,
    )
    results = pd.concat(list(map(compute_interacting_cell_scores_row_partial, gene_pairs)), axis=1)

    metabolite_gene_pair_df = metabolite_gene_pair_df[metabolite_gene_pair_df.gene_pair.isin(results.columns)]

    results_transposed = results.T
    results_transposed.index = ['-'.join('-'.join(elem) if isinstance(elem, list) else elem for elem in tup) for tup in results_transposed.index]
    df_metabolites_grouped = pd.DataFrame(index=results.index)

    for metabolite, group in metabolite_gene_pair_df.groupby('metabolite'):
        gene_pairs = group['gene_pair'].tolist()
        gene_pairs = ['-'.join('-'.join(elem) if isinstance(elem, list) else elem for elem in tup) for tup in gene_pairs]
        df_metabolites_grouped[metabolite] = results_transposed.loc[gene_pairs].sum()

    unique_ind = len(df_metabolites_grouped.index.unique()) == df_metabolites_grouped.shape[0]
    df_metabolites_grouped = df_metabolites_grouped.groupby(df_metabolites_grouped.index).sum() if not unique_ind else df_metabolites_grouped
    
    ct_pair_results_dict = {ct_pair: df_metabolites_grouped}

    return ct_pair_results_dict


def compute_interacting_cell_scores_row(
    gene_pair, ct_pair, counts, weights, cell_types, cells, genes, func,
):

    ct_t, ct_u = ct_pair
    gene_a, gene_b = gene_pair
    gene_a_ind = genes.index(gene_a) if isinstance(gene_a, str) else [genes.index(g_a) for g_a in gene_a]
    gene_b_ind = genes.index(gene_b) if isinstance(gene_b, str) else [genes.index(g_b) for g_b in gene_b]
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
    interacting_cell_scores = pd.DataFrame({'scores': 0}, index = cells.tolist())

    rows, cols = weights_ct.nonzero()
    for i, j in zip(rows, cols):
        w_ij = weights_ct[i, j]

        ai = counts_t[gene_a_ind, i] if isinstance(gene_a, str) else gmean(counts_t[gene_a_ind, i])
        cell_i = cells_t[i]
        bj = counts_u[gene_b_ind, j] if isinstance(gene_b, str) else gmean(counts_u[gene_b_ind, j])
        cell_j = cells_u[j]
        cell_pair = (cell_i, cell_j)

        if ai > 0 or bj > 0:
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
            interacting_cell_scores.loc[cell_i] += w_ij * f_ai_jb
    
    interacting_cell_scores.columns = [gene_pair]

    return interacting_cell_scores


def compute_interacting_cell_scores_no_ct_row(
    gene_pair, counts, weights, cells, genes, func,
):

    gene_a, gene_b = gene_pair
    gene_a_ind = genes.index(gene_a) if isinstance(gene_a, str) else [genes.index(g_a) for g_a in gene_a]
    gene_b_ind = genes.index(gene_b) if isinstance(gene_b, str) else [genes.index(g_b) for g_b in gene_b]

    interacting_cells = {"cells": [], "scores": []}
    interacting_cell_scores = pd.DataFrame({'scores': 0}, index = cells.tolist())

    rows, cols = weights.nonzero()
    for i, j in zip(rows, cols):
        w_ij = weights[i, j]

        ai = counts[gene_a_ind, i] if isinstance(gene_a, str) else gmean(counts[gene_a_ind, i])
        cell_i = cells[i]
        bj = counts[gene_b_ind, j] if isinstance(gene_b, str) else gmean(counts[gene_b_ind, j])
        cell_j = cells[j]
        cell_pair = (cell_i, cell_j)

        if ai > 0 or bj > 0:
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
            interacting_cell_scores.loc[cell_i] += w_ij * f_ai_jb
    
    interacting_cell_scores.columns = [gene_pair]

    return interacting_cell_scores


def compute_CCC_scores(
    counts_1: np.array,
    counts_2: np.array,
    weights: sparse.COO,
):

    if len(weights.shape) == 3:
        scores = np.tensordot(weights, counts_1 * counts_2, axes=([2], [0])).sum(axis=1)
    else:
        scores = (weights @ (counts_1 * counts_2)).sum(axis=0)

    return scores


def get_ct_pair_weights(weights, cell_type_pairs, cell_types):

    w_nrow, w_ncol = weights.shape
    n_ct_pairs = len(cell_type_pairs)

    extract_weights_results = partial(
        extract_ct_pair_weights,
        weights=weights,
        cell_type_pairs=cell_type_pairs,
        cell_types=cell_types,
    )
    results = list(map(extract_weights_results, cell_type_pairs))

    w_new_data_all = [x[0] for x in results]
    w_new_coords_3d_all = [x[1] for x in results]
    w_new_coords_3d_all = np.hstack(w_new_coords_3d_all)
    w_new_data_all = np.concatenate(w_new_data_all)

    weigths_ct_pairs = sparse.COO(w_new_coords_3d_all, w_new_data_all, shape=(n_ct_pairs, w_nrow, w_ncol))

    return weigths_ct_pairs


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


def get_interacting_cell_type_pairs(x, weights, cell_types):
    ct_1, ct_2 = x

    ct_1_bin = cell_types == ct_1
    ct_2_bin = cell_types == ct_2

    weights = weights.tocsc()
    cell_types_weights = weights[ct_1_bin,][:, ct_2_bin]

    return bool(cell_types_weights.nnz)


def conditional_eg2_cellcom(counts, weights):

    counts_sq = counts ** 2

    if len(weights.shape) == 3:
        weights_sq_data = weights.data ** 2
        weights_sq = sparse.COO(weights.coords, weights_sq_data, shape=weights.shape)
        out_eg2 = np.tensordot(counts_sq, weights_sq, axes=([1], [1])).sum(axis=2).todense()
    else:
        weights_sq = weights ** 2
        out_eg2 = (counts_sq @ weights_sq).sum(axis=1)

    return out_eg2


def compute_p_results(ct_pair, cell_type_pairs, gene_pair_cor, gene_pairs_ind, gene_pairs_per_ct_pair_ind, Wtot2, eg2s):
    i = cell_type_pairs.index(ct_pair)
    gene_pair_cor_ct = gene_pair_cor[i, :]
    gene_pairs_ind_ct_pair = gene_pairs_per_ct_pair_ind[
        ct_pair
    ]  # If we consider all the gene pairs (irrespective of the cell type pair) use 'gene_pairs_ind' directly

    C = []
    EG2 = []
    for gene_pair_ind_ct_pair in gene_pairs_ind_ct_pair:
        idx = gene_pairs_ind.index(gene_pair_ind_ct_pair)
        g1_ind, g2_ind = gene_pair_ind_ct_pair
        lc = gene_pair_cor_ct[idx]
        if g1_ind == g2_ind:
            eg2 = Wtot2[i]
        else:
            eg2 = eg2s[g1_ind, i] if type(g1_ind) is not list else np.max(eg2s[g1_ind, i])
        C.append(lc)
        EG2.append(eg2)

    EG = [0 for i in range(len(gene_pairs_ind_ct_pair))]

    stdG = [(EG2[i] - EG[i] ** 2) ** 0.5 for i in range(len(gene_pairs_ind_ct_pair))]
    stdG = [1 if stdG[i] == 0 else stdG[i] for i in range(len(stdG))]

    Z = [(C[i] - EG[i]) / stdG[i] for i in range(len(gene_pairs_ind_ct_pair))]

    return (C, Z)


def compute_p_results_no_ct(gene_pair_cor, gene_pairs_ind, Wtot2, eg2s):

    C = []
    EG2 = []
    for gene_pair_ind in gene_pairs_ind:
        idx = gene_pairs_ind.index(gene_pair_ind)
        g1_ind, g2_ind = gene_pair_ind
        lc = gene_pair_cor[idx]
        if g1_ind == g2_ind:
            eg2 = Wtot2
        else:
            eg2 = eg2s[g1_ind] if type(g1_ind) is not list else np.max(eg2s[g1_ind])
        C.append(lc)
        EG2.append(eg2)

    EG = [0 for i in range(len(gene_pairs_ind))]

    stdG = [(EG2[i] - EG[i] ** 2) ** 0.5 for i in range(len(gene_pairs_ind))]
    stdG = [1 if stdG[i] == 0 else stdG[i] for i in range(len(stdG))]

    Z = [(C[i] - EG[i]) / stdG[i] for i in range(len(gene_pairs_ind))]

    return (C, Z)


def compute_np_results(ct_pair, cell_type_pairs, gene_pair_cor, pvals, gene_pairs_ind, gene_pairs_per_ct_pair_ind):
    i = cell_type_pairs.index(ct_pair)
    gene_pair_cor_ct = gene_pair_cor[i, :]
    pvals_ct = pvals[i, :]
    gene_pairs_ind_ct_pair = gene_pairs_per_ct_pair_ind[
        ct_pair
    ]  # If we consider all the gene pairs (irrespective of the cell type pair) use 'gene_pairs_ind' directly

    C = []
    p_values = []
    for gene_pair_ind_ct_pair in gene_pairs_ind_ct_pair:
        idx = gene_pairs_ind.index(gene_pair_ind_ct_pair)
        lc = gene_pair_cor_ct[idx]
        p_value = pvals_ct[idx]

        C.append(lc.reshape(1))
        p_values.append(p_value.reshape(1))

    C = list(np.concatenate(C))
    p_values = list(np.concatenate(p_values))

    return (C, p_values)


def compute_np_results_no_ct(gene_pair_cor, pvals, gene_pairs_ind):

    C = []
    p_values = []
    for gene_pair_ind in gene_pairs_ind:
        idx = gene_pairs_ind.index(gene_pair_ind)
        lc = gene_pair_cor[idx]
        p_value = pvals[idx]

        C.append(lc.reshape(1))
        p_values.append(p_value.reshape(1))

    C = list(np.concatenate(C))
    p_values = list(np.concatenate(p_values))

    return (C, p_values)


def get_cell_communication_results(
    adata, 
    genes,
    layer_key_p_test,
    layer_key_np_test,
    model, 
    cell_types, 
    cell_type_pairs,
    D,
    test,
):
    
    gene_pairs_per_ct_pair = adata.uns['gene_pairs_per_ct_pair']
    gene_pairs_per_ct_pair_ind = adata.uns['gene_pairs_per_ct_pair_ind']

    cell_com_df = pd.DataFrame.from_dict(gene_pairs_per_ct_pair, orient="index").stack().to_frame().reset_index()
    cell_com_df = cell_com_df.drop(["level_1"], axis=1)
    cell_com_df = cell_com_df.rename(columns={"level_0": "cell_type_pair", 0: "gene_pair"})
    cell_com_df["Cell Type 1"], cell_com_df["Cell Type 2"] = zip(*cell_com_df["cell_type_pair"])
    cell_com_df["Gene 1"], cell_com_df["Gene 2"] = zip(*cell_com_df["gene_pair"])
    cell_com_df = cell_com_df.drop(["cell_type_pair", "gene_pair"], axis=1)

    if test in ["parametric", "both"]:
        c_values = adata.uns['ccc_results']['p']['cs']
        cell_com_df['C_p'] = c_values.flatten()
        z_values = adata.uns['ccc_results']['p']['Z']
        cell_com_df['Z'] = z_values.flatten()
        p_values = adata.uns['ccc_results']['p']['Z_pval']
        cell_com_df['Z_pval'] = p_values.flatten()
        FDR_values = adata.uns['ccc_results']['p']['Z_FDR']
        cell_com_df['Z_FDR'] = FDR_values.flatten()

        counts = counts_from_anndata(adata[:, genes], layer_key_p_test, dense=True)
        num_umi = counts.sum(axis=0)
        counts_std = create_centered_counts(counts, model, num_umi)
        counts_std = np.nan_to_num(counts_std)

        c_values_norm = normalize_values(counts_std, cell_types, cell_type_pairs, gene_pairs_per_ct_pair_ind, c_values, D)
        adata.uns['ccc_results']['p']['cs_norm'] = c_values_norm
        cell_com_df['C_norm_p'] = c_values_norm.flatten()

    if test in ["non-parametric", "both"]:
        c_values = adata.uns['ccc_results']['np']['cs']
        cell_com_df['C_np'] = c_values.flatten()
        p_values = adata.uns['ccc_results']['np']['pval']
        cell_com_df['pval_np'] = p_values.flatten()
        FDR_values = adata.uns['ccc_results']['np']['FDR']
        cell_com_df['FDR_np'] = FDR_values.flatten()

        counts = counts_from_anndata(adata[:, genes], layer_key_np_test, dense=True)
        c_values_norm = normalize_values(counts, cell_types, cell_type_pairs, gene_pairs_per_ct_pair_ind, c_values, D)
        adata.uns['ccc_results']['np']['cs_norm'] = c_values_norm
        cell_com_df['C_norm_np'] = c_values_norm.flatten()

    adata.uns['ccc_results']['cell_com_df'] = cell_com_df
    
    return


def get_cell_communication_results_no_ct(
    adata, 
    genes,
    layer_key_p_test,
    layer_key_np_test,
    model,
    D,
    test,
):
    
    gene_pairs = adata.uns['gene_pairs']
    gene_pairs_ind = adata.uns['gene_pairs_ind']

    cell_com_df = pd.DataFrame(gene_pairs)
    cell_com_df = cell_com_df.rename(columns={0: "Gene 1", 1: "Gene 2"})

    if test in ["parametric", "both"]:
        c_values = adata.uns['ccc_results']['p']['cs']
        cell_com_df['C_p'] = c_values
        z_values = adata.uns['ccc_results']['p']['Z']
        cell_com_df['Z'] = z_values
        p_values = adata.uns['ccc_results']['p']['Z_pval']
        cell_com_df['Z_pval'] = p_values
        FDR_values = adata.uns['ccc_results']['p']['Z_FDR']
        cell_com_df['Z_FDR'] = FDR_values

        counts = counts_from_anndata(adata[:, genes], layer_key_p_test, dense=True)
        num_umi = counts.sum(axis=0)
        counts_std = create_centered_counts(counts, model, num_umi)
        counts_std = np.nan_to_num(counts_std)

        c_values_norm = normalize_values_no_ct(counts_std, gene_pairs_ind, c_values, D)
        adata.uns['ccc_results']['p']['cs_norm'] = c_values_norm
        cell_com_df['C_norm_p'] = c_values_norm

    if test in ["non-parametric", "both"]:
        c_values = adata.uns['ccc_results']['np']['cs']
        cell_com_df['C_np'] = c_values
        p_values = adata.uns['ccc_results']['np']['pval']
        cell_com_df['pval_np'] = p_values
        FDR_values = adata.uns['ccc_results']['np']['FDR']
        cell_com_df['FDR_np'] = FDR_values

        counts = counts_from_anndata(adata[:, genes], layer_key_np_test, dense=True)
        c_values_norm = normalize_values_no_ct(counts, gene_pairs_ind, c_values, D)
        adata.uns['ccc_results']['np']['cs_norm'] = c_values_norm
        cell_com_df['C_norm_np'] = c_values_norm

    adata.uns['ccc_results']['cell_com_df'] = cell_com_df
    
    return


def normalize_values(
    counts,
    cell_types,
    cell_type_pairs,
    gene_pairs_per_ct_pair_ind,
    lcs,
    D,
):
    
    c_values_norm = np.zeros(lcs.shape)
    for i in range(len(cell_type_pairs)):

        ct_pair = cell_type_pairs[i]
        ct_t, ct_u = ct_pair
        cell_type_t_mask = [ct == ct_t for ct in cell_types]
        counts_ct_t = counts[:,cell_type_t_mask]
        D_ct_t = D[i][cell_type_t_mask]
        gene_pairs_ind = gene_pairs_per_ct_pair_ind[ct_pair]
        lc_maxs = compute_max_cs(D_ct_t, counts_ct_t, gene_pairs_ind)
        lc_maxs[lc_maxs == 0] = 1
        c_values_norm[i] = lcs[i] / lc_maxs
        c_values_norm[i][np.isinf(c_values_norm[i])] = 1
    
    return c_values_norm


def normalize_values_no_ct(
    counts,
    gene_pairs_ind,
    lcs,
    D,
):
    
    c_values_norm = np.zeros(len(lcs))
    lc_maxs = compute_max_cs(D, counts, gene_pairs_ind)
    lc_maxs[lc_maxs == 0] = 1
    c_values_norm = lcs / lc_maxs
    c_values_norm[np.isinf(c_values_norm)] = 1
    
    return c_values_norm


def compute_max_cs(node_degrees, counts, gene_pairs_ind):

    result = np.zeros(len(gene_pairs_ind))
    for i, gene_pair_ind in enumerate(gene_pairs_ind):
        vals = counts[gene_pair_ind[0]] if type(gene_pair_ind[0]) is not list else np.mean(counts[gene_pair_ind[0]], axis=0)
        result[i] = compute_max_cs_gp(vals, node_degrees)

    return result


@jit(nopython=True)
def compute_max_cs_gp(vals, node_degrees):
    tot = 0.0

    for i in range(node_degrees.size):
        tot += node_degrees[i] * (vals[i] ** 2)

    return tot / 2


def create_centered_counts(counts, model, num_umi):
    """
    Creates a matrix of centered/standardized counts given
    the selected statistical model
    """
    out = np.zeros_like(counts, dtype="double")

    for i in range(out.shape[0]):
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


@jit(nopython=True)
def compute_local_cov_max(vals, node_degrees):
    tot = 0.0

    for i in range(node_degrees.size):
        tot += node_degrees[i] * (vals[i] ** 2)

    return tot / 2


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
