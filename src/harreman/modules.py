r each module for each gene
        Dimensions are genes x modules

    """

    use_raw = layer_key == "use_raw"
    modules = adata.uns["gene_modules"]

    umi_counts = adata.uns['umi_counts']

    modules_to_compute = sorted([x for x in modules.keys() if x != '-1'])

    print(f"Computing scores for {len(modules_to_compute)} modules...")

    module_scores = {}
    gene_loadings = pd.DataFrame(index=adata.var_names)
    gene_modules = {}
    for module in tqdm(modules_to_compute):
        module_genes = modules[module]

        if method == 'PCA':
            scores, loadings = compute_scores_PCA(
                adata[:, module_genes].copy(),
                layer_key,
                model,
                umi_counts,
                neighbor_smoothing,
            )
        elif method == 'LDVAE':
            scores, loadings = compute_scores_LDVAE(
                adata[:, module_genes].copy(),
            )
        else:
            raise ValueError('Invalid method: Please choose either "PCA" or "LDVAE".')

        module_name = f'HOTSPOT_{module}'
        module_scores[module_name] = scores
        gene_loadings[module_name] = pd.Series(loadings, index=module_genes)
        gene_modules[module_name] = module_genes


    module_scores = pd.DataFrame(module_scores)

    module_scores.index = adata.obs_names if not use_raw else adata.raw.obs.index

    adata.obsm['module_scores'] = module_scores
    adata.varm['gene_loadings'] = gene_loadings
    adata.uns["gene_modules_dict"] = gene_modules


def compute_scores_PCA(
        adata, layer_key, model, num_umi, _lambda=.9):
    """
    counts_sub: row-subset of counts matrix with genes in the module
    """

    weights = adata.obsp['weights']

    counts_sub = counts_from_anndata(adata, layer_key, dense=True)

    cc_smooth = np.zeros_like(counts_sub, dtype=np.float64)

    for i in range(counts_sub.shape[0]):

        counts_row = counts_sub[i, :]
        centered_row = create_centered_counts_row(counts_row, model, num_umi)
        out = weights @ centered_row
        weights_sum = np.array(weights.sum(axis=1).T)[0]
        weights_sum[weights_sum == 0] = 1
        out /= weights_sum
        centered_row = (out * _lambda) + (1 - _lambda) * centered_row
        cc_smooth[i] = centered_row

    pca_data = cc_smooth

    model = PCA(n_components=1)
    scores = model.fit_transform(pca_data.T)
    loadings = model.components_.T

    sign = model.components_.mean()  # may need to flip
    if sign < 0:
        scores = scores * -1
        loadings = loadings * -1

    scores = scores[:, 0]
    loadings = loadings[:, 0]

    return scores, loadings


def compute_scores_LDVAE(
        adata):
    """
    counts_sub: row-subset of counts matrix with genes in the module
    """

    scvi.model.LinearSCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.LinearSCVI(adata, n_latent=1)

    model.train(max_epochs=250, plan_kwargs={"lr": 5e-3}, check_val_every_n_epoch=10)

    Z_hat = model.get_latent_representation()
    Z_hat_list = Z_hat.tolist()

    scores = np.array([x for xs in Z_hat_list for x in xs])

    loadings = model.get_loadings()
    loadings = loadings['Z_0']

    return scores, loadings


def sort_linkage(Z, node_index, node_values):
    """
    Sorts linkage by 'node_values' in place
    """

    N = Z.shape[0] + 1  # number of leaves

    if node_index < 0:
        return

    left_child = int(Z[node_index, 0] - N)
    right_child = int(Z[node_index, 1] - N)

    swap = False

    if left_child < 0 and right_child < 0:
        swap = False
    elif left_child < 0 and right_child >= 0:
        swap = True
    elif left_child >= 0 and right_child < 0:
        swap = False
    else:
        if node_values[left_child] > node_values[right_child]:
            swap = True
        else:
            swap = False

    if swap:
        Z[node_index, 0] = right_child + N
        Z[node_index, 1] = left_child + N

    sort_linkage(Z, left_child, node_values)
    sort_linkage(Z, right_child, node_values)


def calc_mean_dists(Z, node_index, out_mean_dists):
    """
    Calculates the mean density of joins
    for sub-trees underneath each node
    """

    N = Z.shape[0] + 1  # number of leaves

    left_child = int(Z[node_index, 0] - N)
    right_child = int(Z[node_index, 1] - N)

    if left_child < 0:
        left_average = 0
        left_merges = 0
    else:
        left_average, left_merges = calc_mean_dists(
            Z, left_child, out_mean_dists
        )

    if right_child < 0:
        right_average = 0
        right_merges = 0
    else:
        right_average, right_merges = calc_mean_dists(
            Z, right_child, out_mean_dists
        )

    this_height = Z[node_index, 2]
    this_merges = left_merges + right_merges + 1
    this_average = (
        left_average * left_merges + right_average * right_merges + this_height
    ) / this_merges

    out_mean_dists[node_index] = this_average

    return this_average, this_merges


def prop_label(Z, node_index, label, labels, out_clusters):
    """
    Propagates node labels downward if they are not -1
    Used to find the correct cluster label at the leaves
    """

    N = Z.shape[0] + 1  # number of leaves

    if label == -1:
        label = labels[node_index]

    left_child = int(Z[node_index, 0] - N)
    right_child = int(Z[node_index, 1] - N)

    if left_child < 0:
        out_clusters[left_child + N] = label
    else:
        prop_label(Z, left_child, label, labels, out_clusters)

    if right_child < 0:
        out_clusters[right_child + N] = label
    else:
        prop_label(Z, right_child, label, labels, out_clusters)


def prop_label2(Z, node_index, label, labels, out_clusters):
    """
    Propagates node labels downward
    Helper method used in assign_modules
    """

    N = Z.shape[0] + 1  # number of leaves

    parent_label = label
    this_label = labels[node_index]

    if this_label == -1:
        new_label = parent_label
    else:
        new_label = this_label

    left_child = int(Z[node_index, 0] - N)
    right_child = int(Z[node_index, 1] - N)

    if left_child < 0:
        out_clusters[left_child + N] = new_label
    else:
        prop_label2(Z, left_child, new_label, labels, out_clusters)

    if right_child < 0:
        out_clusters[right_child + N] = new_label
    else:
        prop_label2(Z, right_child, new_label, labels, out_clusters)


def assign_modules(Z, leaf_labels, offset, MIN_THRESHOLD=10, Z_THRESHOLD=3):
    clust_i = 0

    labels = np.ones(Z.shape[0])*-1
    N = Z.shape[0]+1

    mean_dists = np.zeros(Z.shape[0])
    calc_mean_dists(Z, Z.shape[0]-1, mean_dists)

    for i in range(Z.shape[0]):

        ca = int(Z[i, 0])
        cb = int(Z[i, 1])

        if ca - N < 0:  # leaf node
            n_members_a = 1
            clust_a = -1
        else:
            n_members_a = Z[ca-N, 3]
            clust_a = labels[ca-N]

        if cb - N < 0:  # leaf node
            n_members_b = 1
            clust_b = -1
        else:
            n_members_b = Z[cb-N, 3]
            clust_b = labels[cb-N]

        if Z[i, 2] > offset - Z_THRESHOLD:
            new_clust_assign = -1
        elif (n_members_a >= MIN_THRESHOLD and n_members_b >= MIN_THRESHOLD):
            # don't join them
            # assign the one with the larger mean distance
            dist_a = mean_dists[ca-N]
            dist_b = mean_dists[cb-N]
            if dist_a >= dist_b:
                new_clust_assign = clust_a
            else:
                new_clust_assign = clust_b
        elif n_members_a >= MIN_THRESHOLD:
            new_clust_assign = clust_a
        elif n_members_b >= MIN_THRESHOLD:
            new_clust_assign = clust_b
        elif (n_members_b + n_members_a) >= MIN_THRESHOLD:
            # A new cluster is born!
            new_clust_assign = clust_i
            clust_i += 1
        else:
            new_clust_assign = -1  # Still too small

        labels[i] = new_clust_assign

    out_clusters = np.ones(N)*-2
    prop_label2(Z, Z.shape[0]-1, labels[-1], labels, out_clusters)

    # remap out_clusters
    unique_clusters = list(np.sort(np.unique(out_clusters)))

    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    clust_map = {
        x: i+1 for i, x in enumerate(unique_clusters)
    }
    clust_map[-1] = -1

    out_clusters = [clust_map[x] for x in out_clusters]
    out_clusters = pd.Series(out_clusters, index=leaf_labels)

    return out_clusters


def assign_modules_core(Z, leaf_labels, offset, MIN_THRESHOLD=10, Z_THRESHOLD=3):
    clust_i = 0

    labels = np.ones(Z.shape[0])*-1
    N = Z.shape[0]+1

    for i in range(Z.shape[0]):

        ca = int(Z[i, 0])
        cb = int(Z[i, 1])

        if ca - N < 0:  # leaf node
            n_members_a = 1
            clust_a = -1
        else:
            n_members_a = Z[ca-N, 3]
            clust_a = labels[ca-N]

        if cb - N < 0:  # leaf node
            n_members_b = 1
            clust_b = -1
        else:
            n_members_b = Z[cb-N, 3]
            clust_b = labels[cb-N]

        if (n_members_a >= MIN_THRESHOLD and n_members_b >= MIN_THRESHOLD):
            # don't join them
            new_clust_assign = -1
        elif Z[i, 2] > offset - Z_THRESHOLD:
            new_clust_assign = -1
        elif n_members_a >= MIN_THRESHOLD:
            new_clust_assign = clust_a
        elif n_members_b >= MIN_THRESHOLD:
            new_clust_assign = clust_b
        elif (n_members_b + n_members_a) >= MIN_THRESHOLD:
            # A new cluster is born!
            new_clust_assign = clust_i
            clust_i += 1
        else:
            new_clust_assign = -1  # Still too small

        labels[i] = new_clust_assign

    out_clusters = np.ones(N)*-2
    prop_label(Z, Z.shape[0]-1, labels[-1], labels, out_clusters)

    # remap out_clusters
    unique_clusters = list(np.sort(np.unique(out_clusters)))

    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    clust_map = {
        x: i+1 for i, x in enumerate(unique_clusters)
    }
    clust_map[-1] = -1

    out_clusters = [clust_map[x] for x in out_clusters]
    out_clusters = pd.Series(out_clusters, index=leaf_labels)

    return out_clusters


def compute_modules(adata, min_gene_threshold=15, fdr_threshold=None, z_threshold=None, core_only=False):
    """
    Assigns modules from the gene pair-wise Z-scores

    Parameters
    ----------
    Z_scores: pandas.DataFrame
        local correlations between genes
    min_gene_threshold: int, optional
        minimum number of genes to create a module
    fdr_threshold: float, optional
        used to determine minimally significant z_score
    core_only: bool, optional
        whether or not to assign unassigned genes to a module

    Returns
    -------
    modules: pandas.Series
        maps gene id to module id
    linkage: numpy.ndarray
        Linkage matrix in the format used by scipy.cluster.hierarchy.linkage

    """

    # Determine Z_Threshold from FDR threshold

    Z_scores = adata.uns["lc_zs"]

    if z_threshold is None:
        allZ = squareform(  # just in case slightly not symmetric
            Z_scores.values/2 + Z_scores.values.T/2
        )
        allZ = np.sort(allZ)
        allP = norm.sf(allZ)
        allP_c = multipletests(allP, method='fdr_bh')[1]
        ii = np.nonzero(allP_c < fdr_threshold)[0]
        if ii.size > 0:
            z_threshold = allZ[ii[0]]
        else:
            z_threshold = allZ[-1]+1

    # Compute the linkage matrix
    dd = Z_scores.copy().values
    np.fill_diagonal(dd, 0)
    condensed = squareform(dd)*-1
    offset = condensed.min() * -1
    condensed += offset
    Z = linkage(condensed, method='average')

    # Linkage -> Modules
    if core_only:
        out_clusters = assign_modules_core(
            Z, offset=offset, MIN_THRESHOLD=min_gene_threshold,
            leaf_labels=Z_scores.index, Z_THRESHOLD=z_threshold)
    else:
        out_clusters = assign_modules(
            Z, offset=offset, MIN_THRESHOLD=min_gene_threshold,
            leaf_labels=Z_scores.index, Z_THRESHOLD=z_threshold)

    # Sort the leaves of the linkage matrix (for plotting)
    mean_dists = np.zeros(Z.shape[0])
    calc_mean_dists(Z, Z.shape[0]-1, mean_dists)
    linkage_out = Z.copy()
    sort_linkage(linkage_out, Z.shape[0]-1, mean_dists)

    out_clusters.name = 'Module'

    gene_modules_dict = {}
    for mod in out_clusters.unique():
        gene_modules_dict[str(mod)] = out_clusters[out_clusters == mod].index.tolist()

    adata.uns["gene_modules"] = gene_modules_dict
    adata.uns["linkage"] = linkage_out


def compute_sig_mod_enrichment(adata, norm_data_key, signature_varm_key):

    use_raw = norm_data_key == "use_raw"
    genes = adata.raw.var.index if use_raw else adata.var_names

    sig_matrix = adata.varm[signature_varm_key] if not use_raw else adata.raw.varm[signature_varm_key]
    gene_modules = adata.uns["gene_modules_dict"]

    signatures = {}

    for signature in sig_matrix.columns:

        if all(x in sig_matrix[signature].unique().tolist() for x in [-1, 1]):
            sig_genes_up = sig_matrix[sig_matrix[signature] == 1].index.tolist()
            sig_genes_down = sig_matrix[sig_matrix[signature] == -1].index.tolist()

            sig_name_up = signature + '_UP'
            sig_name_down = signature + '_DOWN'

            signatures[sig_name_up] = sig_genes_up
            signatures[sig_name_down] = sig_genes_down
        else:
            sig_genes = sig_matrix[sig_matrix[signature] != 0].index.tolist()
            signatures[signature] = sig_genes


    pvals_df = pd.DataFrame(np.nan, index=list(signatures.keys()), columns=list(gene_modules.keys()))
    stats_df = pd.DataFrame(np.nan, index=list(signatures.keys()), columns=list(gene_modules.keys()))

    sig_mod_df = pd.DataFrame(index=genes)

    for signature in signatures.keys():

        sig_genes = signatures[signature]

        for module in gene_modules.keys():

            mod_genes = gene_modules[module]
            sig_mod_genes = list(set(sig_genes) & set(mod_genes))

            M = len(genes)
            n = len(sig_genes)
            N = len(mod_genes)
            x = len(sig_mod_genes)

            pval = hypergeom.sf(x-1, M, n, N)

            if pval < 0.05:
                sig_mod_name = signature + '_OVERLAP_' + module
                sig_mod_df[sig_mod_name] = 0
                sig_mod_df.loc[sig_mod_genes, sig_mod_name] = 1.0

            e_overlap = n*N/M
            stat = np.log2(x/e_overlap) if e_overlap != 0 else 0

            pvals_df.loc[signature, module] = pval
            stats_df.loc[signature, module] = stat

    FDR_values = multipletests(pvals_df.unstack().values, method='fdr_bh')[1]
    FDR_df = pd.Series(FDR_values, index=pvals_df.stack().index).unstack()

    adata.varm['signatures_overlap'] = sig_mod_df

    return pvals_df, stats_df, FDR_df


def compute_sig_mod_correlation(adata, method):

    signatures = adata.obsm['vision_signatures'].columns.tolist()
    modules = adata.obsm['module_scores'].columns.tolist()

    cor_pval_df = pd.DataFrame(index=modules)
    cor_coef_df = pd.DataFrame(index=modules)

    for signature in signatures:

        correlation_values = []
        pvals = []

        for module in modules:

            signature_df = adata.obsm['vision_signatures'][signature]
            module_df = adata.obsm['module_scores'][module]

            if method == 'pearson':
                correlation_value, pval = pearsonr(signature_df, module_df)
            elif method == 'spearman':
                correlation_value, pval = spearmanr(signature_df, module_df)

            correlation_values.append(correlation_value)
            pvals.append(pval)

        cor_coef_df[signature] = correlation_values
        cor_pval_df[signature] = pvals

    cor_FDR_values = multipletests(cor_pval_df.unstack().values, method='fdr_bh')[1]
    cor_FDR_df = pd.Series(cor_FDR_values, index=cor_pval_df.stack().index).unstack()

    return cor_coef_df, cor_pval_df, cor_FDR_df
