import itertools
import warnings
import time
from functools import partial
from math import ceil
from re import compile, match
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import sparse
from anndata import AnnData
from numba import jit
from pynndescent import NNDescent
from scipy.sparse import csr_matrix, triu, lil_matrix, coo_matrix, vstack
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.preprocessing import normalize
from tqdm import tqdm


def compute_knn_graph(
    adata: AnnData,
    compute_neighbors_on_key: Optional[str] = None,
    distances_obsp_key: Optional[str] = None,
    weighted_graph: Optional[bool] = False,
    neighborhood_radius: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    neighborhood_factor: Optional[int] = 3,
    sample_key: Optional[str] = None,
    tree = None,
    verbose: Optional[bool] = False,
):
    """Compute KNN graph.

    Parameters
    ----------
    adata
        AnnData object.
    compute_neighbors_on_key
        Key in `adata.obsm` to use for computing neighbors. If `None`, use neighbors stored in `adata`. If no neighbors have been previously computed an error will be raised.
    distances_obsp_key
        Distances encoding cell-cell similarities directly. Shape is (cells x cells). Input is key in `adata.obsp`.
    weighted_graph
        Whether or not to create a weighted graph.
    neighborhood_radius
        Neighborhood radius.
    n_neighbors
        Neighborhood size.
    neighborhood_factor
        Used when creating a weighted graph.  Sets how quickly weights decay relative to the distances within the neighborhood. The weight for a cell with a distance d will decay as exp(-d^2/D) where D is the distance to the `n_neighbors`/`neighborhood_factor`-th neighbor.
    sample_key
        Sample information in case the data contains different samples or samples from different conditions. Input is key in `adata.obs`.
    tree
        Root tree node. Can be created using ete3.Tree
    verbose : bool, optional (default: False)
        Whether to print progress and status messages.

    """
    start = time.time()
    
    if n_neighbors is None and neighborhood_radius is None:
        raise ValueError(
            "Either 'n_neighbors' or 'neighborhood_radius' needs to be provided."
        )

    if tree is not None:
        try:
            all_leaves = []
            for x in tree:
                if x.is_leaf():
                    all_leaves.append(x.name)
        except:
            raise ValueError("Can't parse supplied tree")

        if len(all_leaves) != adata.shape[0] or len(
            set(all_leaves) & set(adata.obs_names)
        ) != len(all_leaves):
            raise ValueError(
                "Tree leaf labels don't match observations in supplied AnnData"
            )
        
        if weighted_graph:
            raise ValueError(
                "When using `tree` as the metric space, `weighted_graph=True` is not supported"
            )
        tree_neighbors_and_weights(
            adata, tree, n_neighbors=n_neighbors
        )

    if compute_neighbors_on_key is not None:
        if verbose:
            print("Computing the neighborhood graph...")
        compute_neighbors(
            adata=adata,
            compute_neighbors_on_key=compute_neighbors_on_key,
            n_neighbors=n_neighbors,
            neighborhood_radius=neighborhood_radius,
            sample_key=sample_key,
            verbose=verbose,
        )
    else:
        if distances_obsp_key is not None and distances_obsp_key in adata.obsp:
            if verbose:
                print("Computing the neighborhood graph from distances...")
            compute_neighbors_from_distances(
                adata,
                distances_obsp_key,
                n_neighbors,
                sample_key,
                verbose,
            )

    if 'distances' in adata.obsp:
        if verbose:
            print("Computing the weights...")
        compute_weights(
            adata,
            weighted_graph,
            neighborhood_factor,
        )

    if verbose:
        print("Finished computing the KNN graph in %.3f seconds" %(time.time()-start))

    return


def compute_neighbors(
        adata: AnnData,
        compute_neighbors_on_key: str = None,
        n_neighbors: Optional[int] = None,
        neighborhood_radius: Optional[int] = None,
        sample_key: Optional[str] = None,
        verbose: Optional[bool] = False,
) -> None:
    """
    Computes a nearest-neighbors graph on the AnnData object using either
    radius-based or k-nearest neighbors.

    Parameters
    ----------
    adata : AnnData
        Annotated data object (AnnData).
    compute_neighbors_on_key : str, optional
        Key in `adata.obsm` to compute neighbors on (e.g., spatial coordinates or PCA).
        If None, defaults to 'spatial'.
    n_neighbors : int, optional
        Number of nearest neighbors to compute (for kNN graph).
    neighborhood_radius : int, optional
        Radius to use for radius-based graph computation (in Euclidean space).
        Only used if `n_neighbors` is not provided.
    sample_key : str, optional
        Key in `adata.obs` indicating batch/sample identity.
        Ensures neighbors are only computed within each sample.
    verbose : bool, optional (default: False)
        Whether to print progress and status messages.
    """

    # coords = adata.obsm[compute_neighbors_on_key]
    
    # if sample_key is not None:
    #     print("Adding sample-level specificity to the neighborhood graph...")
    #     adata.uns['sample_key'] = sample_key
    #     n_cells = adata.shape[0]
    #     distances = lil_matrix((n_cells, n_cells))
    #     samples = adata.obs[sample_key].unique().tolist()
        
    #     for sample in tqdm(samples):
    #         sample_indices = np.where(adata.obs[sample_key] == sample)[0]
    #         sample_coords = coords[sample_indices]

    #         if n_neighbors is not None:
    #             nbrs = NearestNeighbors(
    #                 n_neighbors=n_neighbors+1,
    #                 algorithm='ball_tree').fit(sample_coords)
    #             sample_distances = nbrs.kneighbors_graph(sample_coords, mode='distance').todok()
    #         elif neighborhood_radius is not None:
    #             coords_tree = KDTree(sample_coords)
    #             sample_distances = coords_tree.sparse_distance_matrix(coords_tree, neighborhood_radius)

    #         for (i, j), sample_distance in sample_distances.items():
    #             distances[sample_indices[i], sample_indices[j]] = sample_distance
    # else:
    #     if n_neighbors is not None:
    #         nbrs = NearestNeighbors(
    #             n_neighbors=n_neighbors+1,
    #             algorithm='ball_tree').fit(coords)
    #         distances = nbrs.kneighbors_graph(coords, mode='distance')

    #     else:
    #         coords_tree = KDTree(coords)
    #         distances = coords_tree.sparse_distance_matrix(coords_tree, neighborhood_radius)

    # if 'deconv_data' in adata.uns and adata.uns['deconv_data'] is True:
    #     spot_diameter = adata.uns['spot_diameter']
    #     barcodes = adata.obs['barcodes'].unique().tolist()
    #     for barcode in barcodes:
    #         barcode_mask = adata.obs['barcodes'] == barcode
    #         barcode_mask = barcode_mask.reset_index(drop=True)
    #         barcode_mask_ind = barcode_mask[barcode_mask].index.tolist()
    #         if len(barcode_mask_ind) == 1:
    #             continue
    #         barcode_mask_ind_perm = list(itertools.permutations(barcode_mask_ind, 2))
    #         barcode_mask_pos = list(zip(*barcode_mask_ind_perm))
    #         distances[barcode_mask_pos[0],barcode_mask_pos[1]] = spot_diameter/2

    # adata.obsp["distances"] = distances.tocsr()
    
    
    # Optimized version
    
    if compute_neighbors_on_key not in adata.obsm:
        raise ValueError(f"{compute_neighbors_on_key} not found in adata.obsm")
    
    coords = adata.obsm[compute_neighbors_on_key]
    n_cells = adata.n_obs
    distances = lil_matrix((n_cells, n_cells))

    if sample_key is not None:
        if verbose:
            print(f"Restricting graph within samples using '{sample_key}'...")
        adata.uns['sample_key'] = sample_key
        samples = adata.obs[sample_key].unique()
        for sample in tqdm(samples):
            sample_mask = adata.obs[sample_key] == sample
            sample_indices = np.where(sample_mask)[0]
            sample_coords = coords[sample_mask]
            if len(sample_indices) == 0:
                continue
            if n_neighbors is not None:
                nn = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(sample_coords)
                dist = nn.kneighbors_graph(sample_coords, mode='distance')
            elif neighborhood_radius is not None:
                dist = radius_neighbors_graph(sample_coords, radius=neighborhood_radius, mode='distance', include_self=False)
            else:
                raise ValueError("Either n_neighbors or neighborhood_radius must be specified.")
            dist = dist.tocoo()
            distances[sample_indices[dist.row], sample_indices[dist.col]] = dist.data
    else:
        if n_neighbors is not None:
            nn = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(coords)
            distances = nn.kneighbors_graph(coords, mode='distance')
        elif neighborhood_radius is not None:
            distances = radius_neighbors_graph(coords, radius=neighborhood_radius, mode='distance', include_self=False)
        else:
            raise ValueError("Either n_neighbors or neighborhood_radius must be specified.")

    # Deconvolution-aware neighborhood
    if adata.uns.get('deconv_data', False):
        if verbose:
            print("Adding intra-spot connections...")
        spot_diameter = adata.uns['spot_diameter']
        idx = adata.obs.groupby('barcodes').indices
        rows, cols = [], []
        for barcode, inds in idx.items():
            if len(inds) < 2:
                continue
            # Efficient pairwise combinations without permutations
            inds = np.array(inds)
            i, j = np.meshgrid(inds, inds, indexing='ij')
            mask = i != j
            rows.extend(i[mask])
            cols.extend(j[mask])
        extra_distances = csr_matrix((np.full(len(rows), spot_diameter / 2), (rows, cols)), shape=(n_cells, n_cells))
        distances += extra_distances

    adata.obsp["distances"] = distances.tocsr()

    return


def compute_neighbors_from_distances(
        adata: AnnData,
        distances_obsp_key: str = "distances",
        sample_key: Optional[str] = None,
        verbose: Optional[bool] = False,
) -> None:
    """
    Builds a neighborhood graph using a precomputed distance matrix.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing a full distance matrix in `obsp`.
    distances_obsp_key : str
        Key in `adata.obsp` with the full distance matrix.
    sample_key : str, optional
        Key in `adata.obs` to enforce neighbors only within samples.
    verbose : bool, optional (default: False)
        Whether to print progress and status messages.
    """
    
    if distances_obsp_key not in adata.obsp:
        raise ValueError(f"{distances_obsp_key} not found in adata.obsp")
    
    distances_raw = adata.obsp[distances_obsp_key]
    distances_raw = csr_matrix(distances_raw) if isinstance(distances_raw, np.ndarray) else distances_raw
    
    n_cells = adata.shape[0]
    distances = lil_matrix((n_cells, n_cells))
    
    # Restrict by sample
    if sample_key is not None:
        if verbose:
            print(f"Restricting graph within samples using '{sample_key}'...")
        adata.uns['sample_key'] = sample_key
        samples = adata.obs[sample_key].unique().tolist()
        for sample in tqdm(adata.obs[sample_key].unique(), desc="Samples"):
            idx = np.where(adata.obs[sample_key] == sample)[0]
            sub_dist = distances_raw[idx, :][:, idx]
            distances[np.ix_(idx, idx)] = sub_dist
    else:
        distances = distances_raw.copy().tolil()

    if adata.uns.get("deconv_data", False) and "barcodes" in adata.obs:
        if verbose:
            print("Adding intra-spot connections...")
        for barcode, inds in adata.obs.groupby("barcodes").indices.items():
            if len(inds) <= 1:
                continue
            for i, j in itertools.permutations(inds, 2):
                distances[i, j] = spot_diameter / 2

    adata.obsp["distances"] = distances.tocsr()

    return


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
    """
    Computes weights on the neighbors based on a
    gaussian kernel and their distances.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing a full distance matrix in `obsp`.
    weighted_graph : bool
        Whether or not to create a weighted graph.
    neighborhood_factor : int
        Used when creating a weighted graph.  Sets how quickly weights decay relative to the distances within the neighborhood. 
        The weight for a cell with a distance d will decay as exp(-d^2/D) where D is the distance to the `n_neighbors`/`neighborhood_factor`-th neighbor.
    """
    
    # distances = sparse.COO.from_scipy_sparse(adata.obsp['distances'])

    # if not weighted_graph:
    #     weights = distances.copy()
    #     weights_mask = distances != 0

    #     weights.data = np.where(weights_mask.data, 1, 0)
    #     weights.coords = weights_mask.coords
    #     adata.obsp["weights"] = weights.tocsr()
    # else:
    #     k_neighbors = [distances[i].nnz for i in range(distances.shape[0])]

    #     if len(np.unique(k_neighbors)) != 1:
    #         radius_ii = [ceil(k_neighbor / neighborhood_factor) for k_neighbor in k_neighbors]
    #         sigma = np.array([[sorted(distances[i].data)[radius_ii[i]-1]] if k_neighbors[i] != 0 else [0] for i in range(distances.shape[0])], dtype=float)

    #     else:
    #         radius_ii = ceil(np.unique(k_neighbors) / neighborhood_factor)
    #         sigma = np.array([[sorted(distances[i].data)[radius_ii-1]] for i in range(distances.shape[0])], dtype=float)

    #     sigma[sigma == 0] = 1

    #     weights = -1 * distances**2 / sigma**2
    #     weights.data = np.exp(weights.data)

    #     weights_csr = weights.tocsr()
    #     weights_norm = normalize(weights_csr, norm="l1", axis=1)
    #     adata.obsp["weights"] = weights_norm

    #     return
    
    # Optimized version
    
    # Load distance matrix and remove diagonal entries
    distances = sparse.COO.from_scipy_sparse(adata.obsp['distances'])
    i, j = distances.coords
    non_diag_mask = i != j
    i, j, data = i[non_diag_mask], j[non_diag_mask], distances.data[non_diag_mask]
    distances = sparse.COO(coords=[i, j], data=data, shape=distances.shape)

    if not weighted_graph:
        # Unweighted: convert all non-zero distances to 1
        weights = sparse.COO(coords=[i, j], data=np.ones_like(data), shape=distances.shape)
        adata.obsp["weights"] = weights.tocsr()
        return
    
    # Weighted: Gaussian kernel
    n_cells = distances.shape[0]
    row_starts = np.searchsorted(i, np.arange(n_cells), side="left")
    row_ends = np.searchsorted(i, np.arange(n_cells) + 1, side="left")

    sigmas = np.ones(n_cells, dtype=float)
    for idx in range(n_cells):
        start, end = row_starts[idx], row_ends[idx]
        row_data = data[start:end]
        if row_data.size == 0:
            continue
        radius_idx = ceil(len(row_data) / neighborhood_factor) - 1
        sigmas[idx] = np.partition(row_data, radius_idx)[radius_idx]

    # Build exp(-d^2 / sigma^2) weights
    sigma_lookup = sigmas[i]  # map sigma per row
    gaussian_weights = np.exp(-1.0 * data**2 / sigma_lookup**2)

    weights = sparse.COO(coords=[i, j], data=gaussian_weights, shape=distances.shape)
    weights_csr = weights.tocsr()
    weights_norm = normalize(weights_csr, norm="l1", axis=1)

    adata.obsp["weights"] = weights_norm
    
    return


def make_weights_non_redundant(weights):

    w_no_redundant = weights.copy()
    
    rows, cols = w_no_redundant.nonzero()
    upper_diag_mask = rows < cols
    upper_rows, upper_cols = rows[upper_diag_mask], cols[upper_diag_mask]

    w_no_redundant[upper_rows, upper_cols] += w_no_redundant[upper_cols, upper_rows]
    w_no_redundant[upper_cols, upper_rows] = 0
    w_no_redundant.eliminate_zeros()

    return w_no_redundant


# @jit(nopython=True)
# def make_weights_non_redundant_orig(neighbors, weights):
#     w_no_redundant = weights.copy()

#     for i in range(neighbors.shape[0]):
#         for k in range(neighbors.shape[1]):
#             j = neighbors[i, k]

#             if j < i:
#                 continue

#             for k2 in range(neighbors.shape[1]):
#                 if neighbors[j, k2] == i:
#                     w_ji = w_no_redundant[j, k2]
#                     w_no_redundant[j, k2] = 0
#                     w_no_redundant[i, k] += w_ji

#     return w_no_redundant


def tree_neighbors_and_weights(adata, tree, n_neighbors):
    """
    Computes nearest neighbors and associated weights for data
    Uses distance along the tree object

    Names of the leaves of the tree must match the columns in counts

    Parameters
    ==========
    adata
        AnnData object.
    tree: ete3.TreeNode
        The root of the tree
    n_neighbors: int
        Number of neighbors to find

    """

    K = n_neighbors
    cell_labels = adata.obs_names

    all_leaves = []
    for x in tree:
        if x.is_leaf():
            all_leaves.append(x)

    all_neighbors = {}

    for leaf in tqdm(all_leaves):
        neighbors = _knn(leaf, K)
        all_neighbors[leaf.name] = neighbors

    cell_ix = {c: i for i, c in enumerate(cell_labels)}

    knn_ix = lil_matrix((len(all_neighbors), len(all_neighbors)), dtype=np.int8)
    for cell in all_neighbors:
        row = cell_ix[cell]
        nn_ix = [cell_ix[x] for x in all_neighbors[cell]]
        knn_ix[row, nn_ix] = 1

    weights = knn_ix.tocsr()

    adata.obsp["weights"] = weights

    return


def _knn(leaf, K):

    dists = _search(leaf, None, 0)
    dists = pd.Series(dists)
    dists = dists + np.random.rand(len(dists)) * .9  # to break ties randomly

    neighbors = dists.sort_values().index[0:K].tolist()

    return neighbors


def _search(current_node, previous_node, distance):

    if current_node.is_root():
        nodes_to_search = current_node.children
    else:
        nodes_to_search = current_node.children + [current_node.up]
    nodes_to_search = [x for x in nodes_to_search if x != previous_node]

    if len(nodes_to_search) == 0:
        return {current_node.name: distance}

    result = {}
    for new_node in nodes_to_search:

        res = _search(new_node, current_node, distance+1)
        for k, v in res.items():
            result[k] = v

    return result


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
def compute_node_degree_orig(neighbors, weights):

    D = np.zeros(neighbors.shape[0])

    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

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


def compute_node_degree_ct_pair(weights, cell_type_pairs, cell_types):

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
