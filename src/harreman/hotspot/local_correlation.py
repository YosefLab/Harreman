from typing import Optional
import time
import numpy as np
import pandas as pd
import sparse
from scipy.sparse import csr_matrix
from anndata import AnnData
from numba import jit, njit
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm

from . import models
from ..preprocessing.anndata import counts_from_anndata
from .local_autocorrelation import compute_local_cov_max
from ..tools.knn import make_weights_non_redundant


def compute_local_correlation(
    adata: AnnData,
    genes: Optional[list] = None,
    permutation_test: Optional[bool] = False,
    M: Optional[int] = 1000,
):

    start = time.time()

    if genes is None:
        gene_autocorrelation_results = adata.uns['gene_autocorrelation_results']
        genes = gene_autocorrelation_results.loc[gene_autocorrelation_results.FDR < 0.05].sort_values('Z', ascending=False).index

    print(f"Computing pair-wise local correlation on {len(genes)} features...")
    
    layer_key = adata.uns['layer_key']
    model = adata.uns['model']
    
    sample_specific = 'sample_key' in adata.uns.keys()

    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)

    weights = adata.obsp['weights']
    num_umi = adata.uns['umi_counts']

    weights = make_weights_non_redundant(weights)

    row_degrees = np.array(weights.sum(axis=1).T)[0]
    col_degrees = np.array(weights.sum(axis=0))[0]
    D = row_degrees + col_degrees
    
    if sample_specific:
        sample_key = adata.uns['sample_key']
        for sample in adata.obs[sample_key].unique().tolist():
            subset = np.where(adata.obs[sample_key] == sample)[0]
            counts[:,subset] = create_centered_counts(counts[:,subset], model, num_umi[subset])
    else:
        counts = create_centered_counts(counts, model, num_umi)

    eg2s = (((weights + weights.T) @ counts.T) ** 2).sum(axis=0)
    # The equation above is equivalent to (((weights @ counts.T) + (counts @ weights).T) ** 2).sum(axis=0)

    lcs = counts @ weights @ counts.T + counts @ weights.T @ counts.T
    lc_zs = compute_cor_Z_scores(lcs, eg2s)
    
    lc_z_pvals = norm.sf(lc_zs)
    
    if permutation_test:
        perm_array = np.zeros((counts.shape[0], counts.shape[0], M)).astype(np.float16)
        for i in tqdm(range(M)):
            idx = np.random.permutation(counts.shape[1])
            counts = counts[:, idx]
            perm_array[:, :, i] = counts @ weights @ counts.T + counts @ weights.T @ counts.T
        
        x = np.sum(perm_array > lcs[:, :, np.newaxis], axis=2)
        lc_perm_pvals = (x + 1) / (M + 1)
        
        lc_perm_pvals = pd.DataFrame(lc_perm_pvals, index=genes, columns=genes)
        adata.uns["lc_perm_pvals"] = lc_perm_pvals
    
    lc_maxs = compute_max_correlation(D, counts)
    lcs = lcs / lc_maxs
    
    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)
    lc_z_pvals = pd.DataFrame(lc_z_pvals, index=genes, columns=genes)
    
    adata.uns["lcs"] = lcs
    adata.uns["lc_zs"] = lc_zs
    adata.uns["lc_z_pvals"] = lc_z_pvals
    
    print("Finished computing pair-wise local correlation in %.3f seconds" %(time.time()-start))

    return


@jit(nopython=True)
def conditional_eg2(x, neighbors, weights):
    """
    Computes EG2 for the conditional correlation
    """
    N = neighbors.shape[0]
    K = neighbors.shape[1]
  
    t1x = np.zeros(N)

    for i in range(N):
        K = len(neighbors[i][~np.isnan(neighbors[i])])
        for k in range(K):
            j = neighbors[i, k]
            j = int(j)

            wij = weights[i, j]
            if wij == 0:
                continue

            t1x[i] += wij*x[j]
            t1x[j] += wij*x[i]

    out_eg2 = (t1x**2).sum()

    return out_eg2


@jit(nopython=True)
def local_cov_pair(x, y, neighbors, weights):
    """Test statistic for local pair-wise autocorrelation"""
    out = 0

    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        if xi == 0 and yi == 0:
            continue
        K = len(neighbors[i][~np.isnan(neighbors[i])])
        for k in range(K):

            j = neighbors[i, k]
            j = int(j)

            w_ij = weights[i, j]

            xj = x[j]
            yj = y[j]

            out += w_ij*(xi*yj + yi*xj)/2

    return out


@jit(nopython=True)
def local_cov_pair_fast(counts, weights):
    """Test statistic for local pair-wise autocorrelation"""
    counts_t = counts.transpose()
    weights_t = weights.transpose()

    lc_1 = sparse.einsum('ik,kl,lj->ij', counts, weights, counts_t)
    lc_2 = sparse.einsum('ik,kl,lj->ij', counts, weights_t, counts_t)
    lc = lc_1 + lc_2

    # lc = counts @ weights @ counts.T + counts @ weights.T @ counts.T

    return lc


def create_centered_counts(counts, model, num_umi):
    """
    Creates a matrix of centered/standardized counts given
    the selected statistical model
    """
    out = np.zeros_like(counts, dtype='double')

    for i in range(out.shape[0]):

        vals_x = counts[i]

        out_x = create_centered_counts_row(
            vals_x, model, num_umi)

        out[i] = out_x

    return out


def create_centered_counts_row(vals_x, model, num_umi):

    if model == 'bernoulli':
        vals_x = (vals_x > 0).astype('double')
        mu_x, var_x, x2_x = models.bernoulli_model(
            vals_x, num_umi)
    elif model == 'danb':
        mu_x, var_x, x2_x = models.danb_model(
            vals_x, num_umi)
    elif model == 'normal':
        mu_x, var_x, x2_x = models.normal_model(
            vals_x, num_umi)
    elif model == 'none':
        mu_x, var_x, x2_x = models.none_model(
            vals_x, num_umi)
    else:
        raise Exception(f"Invalid Model: {model}")

    var_x[var_x == 0] = 1
    out_x = (vals_x-mu_x)/(var_x**0.5)
    out_x[out_x == 0] = 0

    return out_x


@jit(nopython=True)
def _compute_hs_pairs_inner_centered_cond_sym(
    rowpair, counts, neighbors, weights, eg2s
):
    """
    This version assumes that the counts have already been modeled
    and centered
    """
    row_i, row_j = rowpair

    vals_x = counts[row_i]
    vals_y = counts[row_j]

    lc = local_cov_pair(vals_x, vals_y, neighbors, weights)*2

    # Compute xy
    EG, EG2 = 0, eg2s[row_i]

    stdG = (EG2 - EG ** 2) ** 0.5

    Zxy = (lc - EG) / stdG

    # Compute yx
    EG, EG2 = 0, eg2s[row_j]

    stdG = (EG2 - EG ** 2) ** 0.5

    Zyx = (lc - EG) / stdG

    if abs(Zxy) < abs(Zyx):
        Z = Zxy
    else:
        Z = Zyx

    return (lc, Z)


def compute_cor_Z_scores(
    lc, eg2s
):

    EG, EG2 = 0, eg2s
    stdG = (EG2 - EG ** 2) ** 0.5

    Z = (lc - EG) / stdG[:, np.newaxis]

    Zxy = np.tril(Z, k=-1)
    Zyx = np.tril(Z.T, k=-1)

    Z = np.where(np.abs(Zxy) < np.abs(Zyx), Zxy, Zyx)

    i_upper = np.triu_indices(Z.shape[0], k=1)
    Z[i_upper] = Z.T[i_upper]

    return Z


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


def compute_max_correlation(node_degrees, counts):
    """
    For a Genes x Cells count matrix, compute the maximal pair-wise correlation
    between any two genes
    """

    N_GENES = counts.shape[0]

    gene_maxs = np.zeros(N_GENES)
    for i in range(N_GENES):
        gene_maxs[i] = compute_local_cov_max(node_degrees, counts[i])

    result = gene_maxs.reshape((-1, 1)) + gene_maxs.reshape((1, -1))
    result = result / 2
    return result

