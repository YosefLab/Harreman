an'
        ).fit(coords)
        distances = nnbrs.kneighbors_graph(coords, mode='distance')

    else:
        coords_tree = KDTree(coords)
        distances = coords_tree.sparse_distance_matrix(coords_tree, neighborhood_radius)

    if 'deconv_data' in adata.uns and adata.uns['deconv_data'] is True:
        spot_diameter = adata.uns['spot_diameter']
        barcodes = adata.obs['barcodes'].unique().tolist()
        for barcode in barcodes:
            barcode_mask = adata.obs['barcodes'] == barcode
            barcode_mask = barcode_mask.reset_index(drop=True)
            barcode_mask_ind = barcode_mask[barcode_mask].index.tolist()
            if len(barcode_mask_ind) == 1:
                continue
            barcode_mask_ind_perm = list(itertools.permutations(barcode_mask_ind, 2))
            barcode_mask_pos = list(zip(*barcode_mask_ind_perm))
            distances[barcode_mask_pos[0],barcode_mask_pos[1]] = spot_diameter/2

    if sample_key is not None:
        samples = adata.obs[sample_key].unique().tolist()
        sample_pairs = list(itertools.permutations(samples, 2))
        for sample_pair in sample_pairs:
            sample_1, sample_2 = sample_pair
            sample_1_mask = adata.obs[sample_key] == sample_1
            sample_1_mask = sample_1_mask.reset_index(drop=True)
            sample_1_mask_ind = sample_1_mask[sample_1_mask].index.tolist()
            sample_2_mask = adata.obs[sample_key] == sample_2
            sample_2_mask = sample_2_mask.reset_index(drop=True)
            sample_2_mask_ind = sample_2_mask[sample_2_mask].index.tolist()
            subset = distances.tolil()[sample_1_mask_ind,:][:,sample_2_mask_ind]
            if subset.nnz > 0:
                raise ValueError(
                    "The distance between cells from different samples should be 0."
                )

    # adata.obsp["distances"] = sparse.COO.from_scipy_sparse(distances)
    adata.obsp["distances"] = distances


def compute_neighbors_from_distances(
        adata: AnnData,
        distances_obsp_key: str,
        spot_diameter: int,
        sample_key: str,
) -> None:
    """Computes nearest neighbors and associated weights using
    provided distance matrix directly.

    Parameters
    ----------
    distances: pandas.Dataframe num_cells x num_cells

    Returns
    -------
    neighbors:      pandas.Dataframe num_cells x n_neighbors
    weights:  pandas.Dataframe num_cells x n_neighbors

    """
    distances = adata.obsp[distances_obsp_key]

    distances = csr_matrix(distances) if type(distances) is np.array else distances

    if 'deconv_data' in adata.uns and adata.uns['deconv_data'] is True:
        barcodes = adata.obs['barcodes'].unique().tolist()
        for barcode in barcodes:
            barcode_mask = adata.obs['barcodes'] == barcode
            barcode_mask = barcode_mask.reset_index(drop=True)
            barcode_mask_ind = barcode_mask[barcode_mask].index.tolist()
            if len(barcode_mask_ind) == 1:
                continue
            barcode_mask_ind_perm = list(itertools.permutations(barcode_mask_ind, 2))
            barcode_mask_pos = list(zip(*barcode_mask_ind_perm))
            distances[barcode_mask_pos[0],barcode_mask_pos[1]] = spot_diameter/2

    if sample_key is not None:
        samples = adata.obs[sample_key].unique().tolist()
        sample_pairs = list(itertools.permutations(samples, 2))
        for sample_pair in sample_pairs:
            sample_1, sample_2 = sample_pair
            sample_1_mask = adata.obs[sample_key] == sample_1
            sample_1_mask = sample_1_mask.reset_index(drop=True)
            sample_1_mask_ind = sample_1_mask[sample_1_mask].index.tolist()
            sample_2_mask = adata.obs[sample_key] == sample_2
            sample_2_mask = sample_2_mask.reset_index(drop=True)
            sample_2_mask_ind = sample_2_mask[sample_2_mask].index.tolist()
            subset = distances.tolil()[sample_1_mask_ind,:][:,sample_2_mask_ind]
            if subset.nnz > 0:
                raise ValueError(
                    "The distance between cells from different samples should be 0."
                )

    # adata.obsp["distances"] = sparse.COO.from_scipy_sparse(distances)
    adata.obsp["distances"] = distances


def object_scalar(x):
    obj = np.empty((), dtype=object)
    obj[()] = x
    return obj


def filter_distances(dist_row):
    dist_row_dense = dist_row.todense()
    nzidx = np.where(dist_row_dense)
    ranking = np.argsort(dist_row[nzidx])
    result = tuple(np.array(nzidx)[:, ranking]) if len(nzidx[0]) > 0 else np.nan
    out = pd.Series(np.array(dist_row_dense[result].flatten())[0]) if len(nzidx[0]) > 0 else pd.Series(result)
    return out


def get_indexes(dist_row):
    dist_row_dense = dist_row.todense()
    nzidx = np.where(dist_row_dense)
    ranking = np.argsort(dist_row[nzidx])
    result = tuple(np.array(nzidx)[:, ranking]) if len(nzidx[0]) > 0 else np.nan
    out = pd.Series(result[1].flatten()) if len(nzidx[0]) > 0 else pd.Series(result)
    return out


def compute_weights(
        adata: AnnData,
        weighted_graph: bool,
        neighborhood_factor: int,
) -> None:
    """Computes weights on the nearest neighbors based on a
    gaussian kernel and their distances.

    Kernel width is set to the num_neighbors / neighborhood_factor's distance

    distances:  cells x neighbors ndarray
    neighborhood_factor: float

    returns weights:  cells x neighbors ndarray

    """    
    distances = sparse.COO.from_scipy_sparse(adata.obsp['distances'])

    if not weighted_graph:
        weights = distances.copy()
        weights_mask = distances != 0

        weights.data = np.where(weights_mask.data, 1, 0)
        weights.coords = weights_mask.coords
        adata.obsp["weights"] = weights.tocsr()
    else:
        k_neighbors = [distances[i].nnz for i in range(distances.shape[0])]

        if len(np.unique(k_neighbors)) != 1:
            radius_ii = [ceil(k_neighbor / neighborhood_factor) for k_neighbor in k_neighbors]
            sigma = np.array([[sorted(distances[i].data)[radius_ii[i]-1]] if k_neighbors[i] != 0 else [0] for i in range(distances.shape[0])], dtype=float)

        else:
            radius_ii = ceil(np.unique(k_neighbors) / neighborhood_factor)
            sigma = np.array([[sorted(distances[i].data)[radius_ii-1]] for i in range(distances.shape[0])], dtype=float)

        sigma[sigma == 0] = 1

        weights = -1 * distances**2 / sigma**2
        weights.data = np.exp(weights.data)

        weights_csr = weights.tocsr()
        weights_norm = normalize(weights_csr, norm="l1", axis=1)
        # adata.obsp["weights"] = sparse.COO.from_scipy_sparse(weights_norm)
        adata.obsp["weights"] = weights_norm


@jit(nopython=True)
def make_weights_non_redundant(weights):

    w_no_redundant = weights.copy()
    
    for i in range(weights.shape[0]):
        neighbors = np.where(weights[i] != 0)[0]
        for j in neighbors:

            if j < i:
                continue

            if weights[j, i] != 0:
                w_ji = w_no_redundant[j, i]
                w_no_redundant[j, i] = 0
                w_no_redundant[i, j] += w_ji

    return w_no_redundant


# @jit(nopython=True)
def compute_node_degree(weights):

    D = np.zeros(weights.shape[0])

    for i in range(weights.shape[0]):
        # col_indices = weights[i].nonzero()[1]
        col_indices = np.nonzero(weights[i])[0]
        for j in col_indices:

            j = int(j)
            w_ij = weights[i, j]

            D[i] += w_ij
            D[j] += w_ij

    return D


@jit(nopython=True)
def compute_node_degree_ct(weights_ct, cell_type_mask_ind):

    D = np.zeros(weights_ct.shape[0])

    for i in range(weights_ct.shape[0]):
        col_indices_ct_i = np.nonzero(weights_ct[i])[0]
        col_indices_ct_i = [col_index_ct_i for col_index_ct_i in col_indices_ct_i if col_index_ct_i in cell_type_mask_ind]
        if len(col_indices_ct_i) == 0:
            continue
        for j in col_indices_ct_i:
            ind_i = cell_type_mask_ind[i]
            j = int(j)
            w_ij = weights_ct[i, j]

            D[ind_i] += w_ij
            D[j] += w_ij

    return D


# @jit(nopython=True)
def compute_node_degree_ct_pair(weights, cell_type_pairs, cell_types):

    # D = np.zeros(weights_ct.shape[0])

    # for i in range(weights_ct.shape[0]):
    #     col_indices_ct_i = np.nonzero(weights_ct[i])[0]
    #     col_indices_ct_i = [col_index_ct_i for col_index_ct_i in col_indices_ct_i if col_index_ct_i in cell_type_mask_u_ind]
    #     if len(col_indices_ct_i) == 0:
    #         continue
    #     for j in col_indices_ct_i:
    #         ind_i = cell_type_mask_t_ind[i]
    #         j = int(j)
    #         w_ij = weights_ct[i, j]

    #         D[ind_i] += w_ij
    #         D[j] += w_ij

    w_nrow, w_ncol = weights.shape
    n_ct_pairs = len(cell_type_pairs)

    extract_counts_weights_results = partial(
        extract_ct_pair_weights,
        weights=weights,
        cell_type_pairs=cell_type_pairs,
        cell_types=cell_types,
    )
    results = list(map(extract_counts_weights_results, cell_type_pairs))

    w_new_data_all = [x[0] for x in results]
    w_new_coords_3d_all = [x[1] for x in results]
    w_new_coords_3d_all = np.hstack(w_new_coords_3d_all)
    w_new_data_all = np.concatenate(w_new_data_all)

    weigths_ct_pairs = sparse.COO(w_new_coords_3d_all, w_new_data_all, shape=(n_ct_pairs, w_nrow, w_ncol))

    row_degrees = weigths_ct_pairs.sum(axis=2).todense()
    col_degrees = weigths_ct_pairs.sum(axis=1).todense()
    D = row_degrees + col_degrees

    return D


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
