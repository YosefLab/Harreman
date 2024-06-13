(axis=1)].index
    else:
        genes = adata.var_names if not use_raw else adata.raw.var.index

    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)

    weights = adata.obsp['weights']
    # We remove all-zero genes from the counts matrix and we update the gene list
    genes = genes[~np.all(counts == 0, axis=1)]
    counts = counts[~np.all(counts == 0, axis=1)]
    num_umi = np.array(counts.sum(axis=0))

    # D = compute_node_degree(weights)

    row_degrees = weights.sum(axis=1).todense()
    col_degrees = weights.sum(axis=0).todense()
    D = row_degrees + col_degrees

    # Wtot2 = (weights**2).sum()

    weights_sq_data = weights.data ** 2
    weights_sq = sparse.COO(weights.coords, weights_sq_data, shape=weights.shape)
    Wtot2 = weights_sq.data.sum()

    weights_data = weights.data
    weights_coords = weights.coords

    def data_iter():
        for i in range(counts.shape[0]):
            vals = counts[i]
            if issparse(vals):
                vals = np.asarray(vals.A).ravel()
            vals = vals.astype("double")
            yield vals

    def initializer():
        global g_weights_data
        global g_weights_coords
        global g_num_umi
        global g_model
        global g_Wtot2
        global g_D
        g_weights_data = weights_data
        g_weights_coords = weights_coords
        g_num_umi = num_umi
        g_model = model
        g_Wtot2 = Wtot2
        g_D = D

    if jobs > 1:

        with multiprocessing.Pool(processes=jobs, initializer=initializer) as pool:

            results = list(
                tqdm(pool.imap(_map_fun_parallel, data_iter()), total=counts.shape[0])
            )
    else:

        def _map_fun(vals):
            return _compute_hs_inner(
                vals, weights_data, weights_coords, num_umi, model, Wtot2, D
            )

        results = list(tqdm(map(_map_fun, data_iter()), total=counts.shape[0]))

    results = pd.DataFrame(results, index=genes, columns=["G", "EG", "stdG", "Z", "C"])

    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]

    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"

    adata.uns['gene_autocorrelation_results'] = results


def compute_local_autocorrelation_fast(
    adata: AnnData,
    layer_key: Optional[Union[Literal["use_raw"], str]] = None,
    database_varm_key: Optional[str] = None,
    model: Optional[str] = None,
):

    use_raw = layer_key == "use_raw"
    if database_varm_key is not None:
        metab_matrix = adata.varm[database_varm_key] if not use_raw else adata.raw.varm[database_varm_key]
        genes = metab_matrix.loc[(metab_matrix!=0).any(axis=1)].index
    else:
        genes = adata.var_names if not use_raw else adata.raw.var.index

    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)

    weights = adata.obsp['weights']
    genes = genes[~np.all(counts == 0, axis=1)]
    counts = counts[~np.all(counts == 0, axis=1)]
    num_umi = np.array(counts.sum(axis=0))

    adata.uns['umi_counts'] = num_umi

    row_degrees = weights.sum(axis=1).todense()
    col_degrees = weights.sum(axis=0).todense()
    D = row_degrees + col_degrees

    Wtot2 = (weights.data ** 2).sum()

    center_vals_f = lambda x: center_values_total(x, num_umi, model)
    counts = np.apply_along_axis(lambda x: center_vals_f(x)[np.newaxis], 1, counts).squeeze(axis=1)

    results = _compute_hs_inner_fast(counts.T, weights.tocsr(), Wtot2, D)
    results = pd.DataFrame(results, index=["G", "G_max", "EG", "stdG", "Z", "C"], columns=genes).T

    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]

    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"

    results = results[["C", "Z", "Pval", "FDR"]]

    adata.uns['gene_autocorrelation_results'] = results


@jit(nopython=True)
def local_cov_weights(x, weights_data, weights_coords):
    out = 0

    for i in range(len(x)):
        mask_i = weights_coords[0] == i
        indices_i = weights_coords[:,mask_i][1]
        values_i = weights_data[mask_i]
        for k in range(len(indices_i)):
            j = indices_i[k]
            j = int(j)

            w_ij = values_i[k]

            xi = x[i]
            xj = x[j]
            if xi == 0 or xj == 0 or w_ij == 0:
                out += 0
            else:
                out += xi * xj * w_ij

    return out


@jit(nopython=True)
def compute_local_cov_max(vals, node_degrees):
    tot = 0.0

    for i in range(node_degrees.size):
        tot += node_degrees[i] * (vals[i] ** 2)

    return tot / 2


def _compute_hs_inner(vals, weights_data, weights_coords, num_umi, model, Wtot2, D):
    """Note, since this is an inner function, for parallelization to work well
    none of the contents of the function can use MKL or OPENBLAS threads.
    Or else we open too many.  Because of this, some simple numpy operations
    are re-implemented using numba instead as it's difficult to control
    the number of threads in numpy after it's imported.
    """
    if model == "bernoulli":
        vals = (vals > 0).astype("double")
        mu, var, x2 = models.bernoulli_model(vals, num_umi)
    elif model == "danb":
        mu, var, x2 = models.danb_model(vals, num_umi)
    elif model == "normal":
        mu, var, x2 = models.normal_model(vals, num_umi)
    elif model == "none":
        mu, var, x2 = models.none_model(vals, num_umi)
    else:
        raise Exception(f"Invalid Model: {model}")

    vals = center_values(vals, mu, var)

    G = local_cov_weights(vals, weights_data, weights_coords)

    EG, EG2 = 0, Wtot2

    stdG = (EG2 - EG * EG) ** 0.5

    Z = (G - EG) / stdG

    G_max = compute_local_cov_max(vals, D)
    C = (G - EG) / G_max

    return [G, EG, stdG, Z, C]


def center_values_total(vals, num_umi, model):
    """
    Note, since this is an inner function, for parallelization to work well
    none of the contents of the function can use MKL or OPENBLAS threads.
    Or else we open too many.  Because of this, some simple numpy operations
    are re-implemented using numba instead as it's difficult to control
    the number of threads in numpy after it's imported
    """

    if model == "bernoulli":
        vals = (vals > 0).astype("double")
        mu, var, x2 = models.bernoulli_model(vals, num_umi)
    elif model == "danb":
        mu, var, x2 = models.danb_model(vals, num_umi)
    elif model == "normal":
        mu, var, x2 = models.normal_model(vals, num_umi)
    elif model == "none":
        mu, var, x2 = models.none_model(vals, num_umi)
    else:
        raise Exception(f"Invalid Model: {model}")

    centered_vals = center_values(vals, mu, var)

    return centered_vals


def _compute_hs_inner_fast(counts, weights, Wtot2, D):

    G = (weights @ counts * counts).sum(axis=0)

    EG, EG2 = 0, Wtot2

    stdG = (EG2 - EG * EG) ** 0.5

    Z = [(G[i] - EG) / stdG for i in range(len(G))]

    G_max = np.apply_along_axis(compute_local_cov_max, 0, counts, D)

    C = (G - EG) / G_max

    EG = [EG for i in range(len(G))]
    stdG = [stdG for i in range(len(G))]

    return [G, G_max, EG, stdG, Z, C]


def _map_fun_parallel(vals):
    global g_weights_data
    global g_weights_coords
    global g_num_umi
    global g_model
    global g_Wtot2
    global g_D
    return _compute_hs_inner(
        vals, g_weights_data, g_weights_coords, g_num_umi, g_model, g_Wtot2, g_D
    )


def compute_communication_autocorrelation(adata, spatial_coords_obsm_key):
    """Computes Geary's C for numerical data."""
    
    metab_scores_df = adata.obsm["metabolite_scores"]
    gene_pair_scores_df = adata.obsm["gene_pair_scores"]

    # Compute autocorrelation on the metabolite scores
    
    metab_adata = AnnData(metab_scores_df)
    metab_adata.obsm[spatial_coords_obsm_key] = adata.obsm[spatial_coords_obsm_key]

    metab_adata.obsm['neighbors_sort'] = adata.obsm['neighbors_sort']
    metab_adata.obsp['weights'] = adata.obsp['weights']

    compute_local_autocorrelation(
        metab_adata,
        model = 'none',
        jobs = 1,
    )

    adata.uns['metabolite_autocorrelation_results'] = metab_adata.uns['gene_autocorrelation_results']

    # Compute autocorrelation on the gene pair scores
    
    gene_pair_adata = AnnData(gene_pair_scores_df)
    gene_pair_adata.obsm[spatial_coords_obsm_key] = adata.obsm[spatial_coords_obsm_key]

    gene_pair_adata.obsm['neighbors_sort'] = adata.obsm['neighbors_sort']
    gene_pair_adata.obsp['weights'] = adata.obsp['weights']

    compute_local_autocorrelation(
        gene_pair_adata,
        model = 'none',
        jobs = 1,
    )

