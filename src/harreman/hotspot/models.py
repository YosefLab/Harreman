from numba import jit
import pandas as pd
import numpy as np
from numba import njit


def danb_model(gene_counts, umi_counts):

    tj = gene_counts.sum()
    tis = umi_counts
    total = tis.sum()

    N = gene_counts.size

    min_size = 10**(-10)

    mu = tj*tis/total
    vv = (gene_counts - mu).var()*(N/(N-1))
    my_rowvar = vv

    # size = (tj**2) * (tis**2).sum()/total**2 / ((N-1)*my_rowvar-tj)
    # regroup terms to protect against overflow errors
    size = ((tj**2) / total) * ((tis**2).sum() / total) / ((N-1)*my_rowvar-tj)

    if size < 0:    # Can't have negative dispersion
        size = 1e9

    if size < min_size and size >= 0:
        size = min_size

    var = mu*(1+mu/size)
    x2 = var+mu**2

    return mu, var, x2


def ct_danb_model(gene_counts, umi_counts, cell_types):

    mu_ct = np.zeros(len(cell_types))
    var_ct = np.zeros(len(cell_types))
    x2_ct = np.zeros(len(cell_types))
    
    for cell_type in np.unique(cell_types):

        gene_counts_ct = gene_counts[cell_types == cell_type]
        umi_counts_ct = umi_counts[cell_types == cell_type]
    
        tj = gene_counts_ct.sum()
        tis = umi_counts_ct
        total = tis.sum()

        N = gene_counts_ct.size

        min_size = 10**(-10)

        mu = tj*tis/total
        vv = (gene_counts_ct - mu).var()*(N/(N-1))
        my_rowvar = vv

        size = ((tj**2) / total) * ((tis**2).sum() / total) / ((N-1)*my_rowvar-tj)

        if size < 0:    # Can't have negative dispersion
            size = 1e9

        if size < min_size and size >= 0:
            size = min_size

        var = mu*(1+mu/size)
        x2 = var+mu**2

        mu_ct[cell_types == cell_type] = mu
        var_ct[cell_types == cell_type] = var
        x2_ct[cell_types == cell_type] = x2

    return mu_ct, var_ct, x2_ct


N_BIN_TARGET = 30


@jit(nopython=True)
def find_gene_p(num_umi, D):
    """
    Finds gene_p such that sum of expected detects
    matches our data

    Performs a binary search on p in the space of log(p)
    """

    low = 1e-12
    high = 1

    if D == 0:
        return 0

    for ITER in range(40):

        attempt = (high*low)**0.5
        tot = 0

        for i in range(len(num_umi)):
            tot = tot + 1-(1-attempt)**num_umi[i]

        if abs(tot-D)/D < 1e-3:
            break

        if tot > D:
            high = attempt
        else:
            low = attempt

    return (high*low)**0.5


def bernoulli_model_scaled(gene_detects, umi_counts):

    D = gene_detects.sum()

    gene_p = find_gene_p(umi_counts, D)

    detect_p = 1-(1-gene_p)**umi_counts

    mu = detect_p
    var = detect_p * (1 - detect_p)
    x2 = detect_p

    return mu, var, x2


def true_params_scaled(gene_p, umi_counts):

    detect_p = 1-(1-gene_p/10000)**umi_counts

    mu = detect_p
    var = detect_p * (1 - detect_p)
    x2 = detect_p

    return mu, var, x2


def bernoulli_model_linear(gene_detects, umi_counts):

    # We modify the 0 UMI counts to 0.001 to remove the NaN values from the qcut output.
    # umi_counts[umi_counts == 0] = 0.01
    
    umi_count_bins, bins = pd.qcut(
        np.log10(umi_counts), N_BIN_TARGET, labels=False, retbins=True,
        duplicates='drop'
    )
    bin_centers = np.array(
        [bins[i] / 2 + bins[i + 1] / 2 for i in range(len(bins) - 1)]
    )

    N_BIN = len(bin_centers)

    bin_detects = bin_gene_detection(gene_detects, umi_count_bins, N_BIN)

    lbin_detects = logit(bin_detects)

    X = np.ones((N_BIN, 2))
    X[:, 1] = bin_centers
    Y = lbin_detects

    b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    detect_p = ilogit(b[0] + b[1] * np.log10(umi_counts))

    mu = detect_p
    var = detect_p * (1 - detect_p)
    x2 = detect_p

    return mu, var, x2


bernoulli_model = bernoulli_model_linear


@njit
def logit(p):
    return np.log(p / (1 - p))


@njit
def ilogit(q):
    return np.exp(q) / (1 + np.exp(q))


@njit
def bin_gene_detection(gene_detects, umi_count_bins, N_BIN):
    bin_detects = np.zeros(N_BIN)
    bin_totals = np.zeros(N_BIN)

    for i in range(len(gene_detects)):
        x = gene_detects[i]
        bin_i = umi_count_bins[i]
        bin_detects[bin_i] += x
        bin_totals[bin_i] += 1

    # Need to account for 0% detects
    #    Add 1 to numerator and denominator
    # Need to account for 100% detects
    #    Add 1 to denominator

    return (bin_detects+1) / (bin_totals+2)


def normal_model(gene_counts, umi_counts):

    """
    Simplest Model - just assumes expression data is normal
    UMI counts are regressed out
    """

    X = np.vstack((np.ones(len(umi_counts)), umi_counts)).T
    y = gene_counts.reshape((-1, 1))

    if umi_counts.var() == 0:

        mu = gene_counts.mean()
        var = gene_counts.var()
        mu = np.repeat(mu, len(umi_counts))
        var = np.repeat(var, len(umi_counts))
        x2 = mu**2 + var

        return mu, var, x2

    B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    mu = X.dot(B)

    var = (y - mu).var()
    var = np.repeat(var, len(umi_counts))

    mu = mu.ravel()

    x2 = mu**2 + var

    return mu, var, x2


def none_model(gene_counts, umi_counts):

    N = gene_counts.size

    mu = np.zeros(N)
    var = np.ones(N)
    x2 = np.ones(N)

    return mu, var, x2
