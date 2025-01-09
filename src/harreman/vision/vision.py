from typing import Literal, Optional, Union

import anndata

from .signature import compute_signatures_anndata
from ..tools.knn import compute_knn_graph
from .VISION_accessor import VISION

data_accessor = VISION()


def analyze_vision(
    adata: Union[str, anndata.AnnData],
    norm_data_key: Optional[Union[Literal["use_raw"], str]] = None,
    compute_neighbors_on_key: Optional[str] = None,
    distances_obsp_key: Optional[str] = None,
    signature_varm_key: Optional[str] = None,
    signature_names_uns_key: Optional[str] = None,
    weighted_graph: Optional[bool] = False,
    neighborhood_radius: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    neighborhood_factor: Optional[int] = 3,
    sample_key: Optional[str] = None,
    obs_df_scores: Optional[bool] = False,
    one_vs_all_obs_cols: Optional[bool] = False,
    one_vs_all_signatures: Optional[bool] = False,
    gene_score_per_signature: Optional[bool] = False,
    scores_only: Optional[bool] = False,
    tree = None,
):
    """Analyze VISION.

    Parameters
    ----------
    adata
        AnnData object.
    norm_data_key
        Key for layer with log library size normalized data. If `None` (default), uses `adata.X`.
    compute_neighbors_on_key
        Key in `adata.obsm` to use for computing neighbors. If `None`, use neighbors stored in `adata`. If no neighbors have been previously computed an error will be raised.
    distances_obsp_key
        Distances encoding cell-cell similarities directly. Shape is (cells x cells). Input is key in `adata.obsp`.
    signature_varm_key
        Key in `adata.varm` for signatures. If `None` (default), no signatures. Matrix should encode positive genes with 1, negative genes with -1, and all other genes with 0.
    signature_names_uns_key
        Key in `adata.uns` for signature names. If `None`, attempts to read columns if `signature_varm_key` is a pandas DataFrame. Otherwise, uses `Signature_1`, `Signature_2`, etc.
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
    obs_df_scores
        Boolean variable indicating whether to compute observation scores or not.
    one_vs_all_obs_cols
        Boolean variable indicating whether to compute one vs all DE analysis of the numerical variables for every categorical variable or not.
    one_vs_all_signatures
        Boolean variable indicating whether to compute one vs all DE analysis of the signature scores for every categorical variable or not.
    gene_score_per_signature
        Boolean variable indicating whether to compute the correlation between gene expression and signature scores for every gene-signature pair or not.

    """
    if isinstance(adata, str):
        adata = anndata.read(str)

    if scores_only is False:
        compute_knn_graph(adata=adata,
                          compute_neighbors_on_key=compute_neighbors_on_key,
                          distances_obsp_key=distances_obsp_key,
                          weighted_graph=weighted_graph,
                          neighborhood_radius=neighborhood_radius,
                          n_neighbors=n_neighbors,
                          neighborhood_factor=neighborhood_factor,
                          sample_key=sample_key,
                          tree=tree)

    data_accessor.adata = adata
    data_accessor.norm_data_key = norm_data_key
    data_accessor.compute_neighbors_on_key = compute_neighbors_on_key
    data_accessor.distances_obsp_key = distances_obsp_key
    data_accessor.signature_varm_key = signature_varm_key
    data_accessor.signature_names_uns_key = signature_names_uns_key
    data_accessor.weighted_graph = weighted_graph
    data_accessor.neighborhood_radius = neighborhood_radius
    data_accessor.n_neighbors = n_neighbors
    data_accessor.neighborhood_factor = neighborhood_factor
    data_accessor.sample_key = sample_key

    if obs_df_scores is True:
        data_accessor.compute_obs_df_scores()

    if one_vs_all_obs_cols is True:
        data_accessor.compute_one_vs_all_obs_cols()

    # compute signatures
    if signature_varm_key is not None:
        adata.obsm["vision_signatures"] = compute_signatures_anndata(
            adata,
            norm_data_key,
            signature_varm_key,
            signature_names_uns_key,
        )

        if scores_only is False:
            data_accessor.compute_signature_scores()

        if one_vs_all_signatures is True:
            data_accessor.compute_one_vs_all_signatures()

    if gene_score_per_signature is True:
        data_accessor.compute_gene_score_per_signature()
