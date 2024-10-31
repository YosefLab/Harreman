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
from scipy.sparse import csr_matrix, triu, lil_matrix
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm


def compute_knn_graph(
    adata: Union[str, AnnData],
    compute_neighbors_on_key: Optional[str] = None,
    distances_obsp_key: Optional[str] = None,
    weighted_graph: Optional[bool] = False,
    neighborhood_radius: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    neighborhood_factor: Optional[int] = 3,
    sample_key: Optional[str] = None,
    tree = None,
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

    """
    start = time.time()

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
        print("Computing the neighborhood graph...")
        compute_neighbors(
            adata=adata,
            compute_neighbors_on_key=compute_neighbors_on_key,
            n_neighbors=n_neighbors,
            neighborhood_radius=neighborhood_radius,
            sample_key=sample_key,
        )
    else:
        if distances_obsp_key is not None and distances_obsp_key in adata.obsp:
            print("Computing the neighborhood graph from distances...")
            compute_neighbors_from_distances(
                adata,
                distances_obsp_key,
                n_neighbors,
                sample_key,
            )

    if 'weights' not in adata.obsp and 'distances' in adata.obsp:
        print("Computing the weights...")
        compute_weights(
            adata,
            weighted_graph,
            neighborhood_factor,
        )

    print("Finished computing the KNN graph in %.3f seconds" %(time.time()-start))

    return


def compute_neighbors(
        adata: Union[str, AnnData],
        compute_neighbors_on_key: Optional[str] = None,
        n_neighbors: Optional[int] = None,
        neighborhood_radius: Optional[int] = None,
        sample_key: Optional[str] = None,
) -> None:

    coords = adata.obsm[compute_neighbors_on_key]

    if n_neighbors is not None:
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors+1,
            algorithm='ball_tree').fit(coords)
        distances = nbrs.kneighbors_graph(coords, mode='distance')

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
        for sample in samples:
            sample_mask = adata.obs[sample_key] == sample
            sample_mask = sample_mask.reset_index(drop=True)
            sample_mask_ind = sample_mask[sample_mask].index.tolist()
            not_sample_mask = adata.obs[sample_key] != sample
            not_sample_mask = not_sample_mask.reset_index(drop=True)
            not_sample_mask_ind = not_sample_mask[not_sample_mask].index.tolist()
            subset = distances.tolil()[sample_mask_ind,:][:,not_sample_mask_ind]
            if subset.nnz > 0:
                raise ValueError(
                    "The distance between cells from different samples should be 0."
                )

    adata.obsp["distances"] = distances

    return


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

    adata.obsp["distances"] = distances

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


@jit(nopython=True)
def make_weights_non_redundant_orig(neighbors, weights):
    w_no_redundant = weights.copy()

    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):
            j = neighbors[i, k]

            if j < i:
                continue

            for k2 in range(neighbors.shape[1]):
                if neighbors[j, k2] == i:
                    w_ji = w_no_redundant[j, k2]
                    w_no_redundant[j, k2] = 0
                    w_no_redundant[i, k] += w_ji

    return w_no_redundant


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
