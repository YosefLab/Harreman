from typing import Literal, Optional, Sequence, Union
import time
from anndata import AnnData
import numpy as np
import pandas as pd
from numba import jit, njit
from scipy.stats import wilcoxon, mannwhitneyu, ranksums
from statsmodels.stats.multitest import multipletests

from ..preprocessing.anndata import counts_from_anndata
from ..hotspot.local_autocorrelation import compute_local_autocorrelation


def load_metabolic_genes(
    species: Optional[Union[Literal["mouse"], Literal["human"]]] = None,
):

    metabolic_genes_paths = {
        'human': "/home/labs/nyosef/oier/Compass_data/metabolic_genes/metabolic_genes_h.csv",
        'mouse': "/home/labs/nyosef/oier/Compass_data/metabolic_genes/metabolic_genes_m.csv"
    }

    metabolic_genes = list(pd.read_csv(metabolic_genes_paths[species], index_col=0, header=None).index)

    return metabolic_genes


@njit
def center_values(vals, mu, var):
    out = np.zeros_like(vals)

    for i in range(len(vals)):
        std = var[i]**0.5
        if std == 0:
            out[i] = 0
        else:
            out[i] = (vals[i] - mu[i])/std

    return out


@njit
def neighbor_smoothing(vals, weights, _lambda=.9):
    """

    output is (neighborhood average) * _lambda + self * (1-_lambda)


    vals: expression matrix (genes x cells)
    weights: neighbor weights (cells x cells)
    _lambda: ratio controlling self vs. neighborhood
    """

    out = np.zeros_like(vals, dtype=np.float64)

    G = vals.shape[0]       # Genes

    for g in range(G):

        row_vals = vals[g, :]
        smooth_row_vals = neighbor_smoothing_row(
            row_vals, weights, _lambda)

        out[g, :] = smooth_row_vals

    return out


@njit
def neighbor_smoothing_row(vals, weights, _lambda=.9):
    """

    output is (neighborhood average) * _lambda + self * (1-_lambda)


    vals: expression matrix (genes x cells)
    neighbors: neighbor indices (cells x K)
    weights: neighbor weights (cells x K)
    _lambda: ratio controlling self vs. neighborhood
    """

    out = np.zeros_like(vals, dtype=np.float64)
    out_denom = np.zeros_like(vals, dtype=np.float64)

    for i in range(weights.shape[0]):

        col_indices = np.nonzero(weights[i])[0]
        xi = vals[i]

        for j in col_indices:

            j = int(j)

            wij = weights[i, j]
            xj = vals[j]

            out[i] += xj*wij
            out[j] += xi*wij

            out_denom[i] += wij
            out_denom[j] += wij

    out /= out_denom

    out = (out * _lambda) + (1 - _lambda) * vals

    return out


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


def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


def get_interacting_cell_type_pairs(x, weights, cell_types):
    ct_1, ct_2 = x

    ct_1_bin = cell_types == ct_1
    ct_2_bin = cell_types == ct_2

    weights = weights.tocsc()
    cell_types_weights = weights[ct_1_bin,][:, ct_2_bin]

    return bool(cell_types_weights.nnz)
