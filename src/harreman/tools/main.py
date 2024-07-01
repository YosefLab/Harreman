import itertools
import time
from random import sample
from typing import Literal, Optional, Sequence, Union

import anndata
from numba import jit
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import sparse
from scipy.sparse import csr_matrix
from scipy.stats.mstats import gmean
from functools import partial
from threadpoolctl import threadpool_limits

from ..preprocessing.anndata import setup_anndata
from .database import counts_from_anndata, extract_lr, extract_transporter_info
from .knn import (
    compute_neighbors,
    compute_neighbors_from_distances,
    compute_node_degree_ct_pair,
    compute_weights,
    make_weights_non_redundant,
)
from .local_autocorrelation import _compute_hs_inner_fast, center_values_total
from .local_correlation import (
    _compute_hs_pairs_inner_centered_cond_sym_fast,
    compute_local_cov_pairs_max,
    create_centered_counts,
)
from .modules import (
    assign_modules,
    assign_modules_core,
    calc_mean_dists,
    compute_scores_LDVAE,
    compute_scores_PCA,
    compute_sig_mod_correlation,
    compute_sig_mod_enrichment,
    sort_linkage,
)
from .signature import compute_signatures_anndata, read_gmt
from ..preprocessing.utils import filter_genes, get_interacting_cell_type_pairs


def signatures_from_gmt(
    adata: AnnData,
    use_raw: bool = False,
    species: Optional[Union[Literal["mouse"], Literal["human"]]] = None,
    gmt_files: Optional[Sequence[str]] = None,
    sig_names: Optional[Sequence[str]] = None,
):
    """Compute signature scores from .gmt files.

    Parameters
    ----------
    adata
        AnnData object to compute signatures for.
    use_raw
        Whether to use adata.raw.X for signature computation.
    species
        Species identity to select one (or more) of the signatures downloaded from MSigDB.
    gmt_files
        List of .gmt files to use for signature computation.
    sig_names
        List of signature file names downloaded from MSigDB to use for signature computation.

    Returns
    -------
    Genes by signatures dataframe and cells by signatures dataframe
    with scores. Index is aligned to genes from adata.

    """
    DOWN_SIG_KEY = "DN"
    UP_SIG_KEY = "UP"

    signature_paths = {
        'human': {
            'BioCarta_Human': 'https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c2.cp.biocarta.v2023.2.Hs.symbols.gmt',
            'Reactome_Human': 'https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c2.cp.reactome.v2023.2.Hs.symbols.gmt',
            'KEGG_Human': 'https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt',
            'GO:BP_Human': 'https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c5.go.bp.v2023.2.Hs.symbols.gmt',
        },
        'mouse': {
            'BioCarta_Mouse': 'https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Mm/m2.cp.biocarta.v2023.2.Mm.symbols.gmt',
            'Reactome_Mouse': 'https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Mm/m2.cp.reactome.v2023.2.Mm.symbols.gmt',
            'GO:BP_Mouse': 'https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Mm/m5.go.bp.v2023.2.Mm.symbols.gmt',
        }
    }

    if species not in ['human', 'mouse'] and gmt_files is None:
        raise ValueError(f'species type: {species} is not supported currently. You should choose either "human" or "mouse".')

    if gmt_files is None and (sig_names is None or sig_names not in list(signature_paths[species].keys())):
        raise ValueError(f'Please provide either your own signature list or select one (or more) of: {list(signature_paths[species].keys())}')

    if type(sig_names) is list:
        if type(gmt_files) is list:
            for sig_name in sig_names:
                if sig_name not in list(signature_paths[species].keys()):
                    raise ValueError(f'{sig_name} is not available. Please select one (or more) of: {list(signature_paths[species].keys())}')
                else:
                    sig_name_path = signature_paths[species][sig_name]
                    gmt_files.append(sig_name_path)
        else:
            gmt_files = []
            for sig_name in sig_names:
                sig_name_path = signature_paths[species][sig_name]
                gmt_files.append(sig_name_path)

    sig_dict = {}
    for gmt_file in gmt_files:
        sig_dict.update(read_gmt(gmt_file))
    index = adata.raw.var.index if use_raw else adata.var_names
    columns = list(sig_dict.keys())
    data = np.zeros((len(index), len(columns)))
    sig_df = pd.DataFrame(index=index, columns=columns, data=data)
    sig_df.index = sig_df.index.str.lower()
    for sig_name, genes_up_down in sig_dict.items():
        for key in [UP_SIG_KEY, DOWN_SIG_KEY]:
            genes = genes_up_down.get(key, None)
            if genes is not None:
                genes = pd.Index(genes).str.lower()
                genes = genes.intersection(sig_df.index)
                sig_df.loc[genes, sig_name] = 1.0 if key == UP_SIG_KEY else -1.0
    sig_df.index = index
    sig_df = sig_df.loc[:, (sig_df!=0).any(axis=0)]

    adata.varm["signatures"] = sig_df

    return


def extract_interaction_db(
    adata: AnnData,
    use_raw: bool = False,
    species: Optional[Union[Literal["mouse"], Literal["human"]]] = None,
    database: Optional[Union[Literal["transporter"], Literal["LR"], Literal["both"]]] = None,
    min_cell: Optional[int] = 0,
):
    """Extract the metabolite transporter or LR database from .csv files.

    Parameters
    ----------
    adata
        AnnData object to compute database for.
    use_raw
        Whether to use adata.raw.X for database computation.
    species
        Species identity to select the LR database from CellChatDB.
    csv_files
        List of .csv files to use for database computation.

    Returns
    -------
    Genes by metabolites (or LRs) dataframe. Index is aligned to genes from adata.

    """
    IMPORT_METAB_KEY = "IMPORT"
    EXPORT_METAB_KEY = "EXPORT"
    BOTH_METAB_KEY = "BOTH"

    if species not in ['human', 'mouse']:
        raise ValueError(f'species type: {species} is not supported currently. You should choose either "human" or "mouse".')

    if database is None:
        raise ValueError('Please one of the options to extract the interaction database: "transporter", "LR" or "both".')

    if database == 'both' or database == 'LR':
        extract_lr(adata, species, min_cell=min_cell)
        index = adata.raw.var.index if use_raw else adata.var_names
        columns = adata.uns['LR_database'].index
        data = np.zeros((len(index), len(columns)))
        LR_df = pd.DataFrame(index=index, columns=columns, data=data)
        LR_df.index = LR_df.index.str.lower()
        for LR_name in columns: #'EFNA5_EPHA3'
            for key in ['ligand', 'receptor']:
                genes = adata.uns[key].loc[LR_name].dropna().values.tolist()
                if len(genes) > 0:
                    genes = pd.Index(genes).str.lower()
                    genes = genes.intersection(LR_df.index)
                    LR_df.loc[genes, LR_name] = 1.0 if key == 'ligand' else -1.0
        LR_df.index = index
        LR_df = LR_df.loc[:, (LR_df!=0).any(axis=0)]

    if database == 'both' or database == 'transporter':
        # Modify the function such that the metabolite DB is provided with the code and not by the user. Save the DB in adata.uns
        metab_dict = {}
        metab_dict = extract_transporter_info(adata, species)
        index = adata.raw.var.index if use_raw else adata.var_names
        columns = list(metab_dict.keys())
        data = np.zeros((len(index), len(columns)))
        metab_df = pd.DataFrame(index=index, columns=columns, data=data)
        metab_df.index = metab_df.index.str.lower()
        for metab_name, genes_dir in metab_dict.items():
            for key in [EXPORT_METAB_KEY, IMPORT_METAB_KEY, BOTH_METAB_KEY]:
                genes = genes_dir.get(key, None)
                if genes is not None:
                    genes = pd.Index(genes).str.lower()
                    genes = genes.intersection(metab_df.index)
                    metab_df.loc[genes, metab_name] = 1.0 if key == EXPORT_METAB_KEY else -1.0 if key == IMPORT_METAB_KEY else 2.0
        metab_df.index = index
        metab_df = metab_df.loc[:, (metab_df!=0).any(axis=0)]

    adata.uns['database_varm_key'] = "database"
    adata.uns['database'] = database

    if database == 'both':
        adata.varm["database"] = pd.concat([LR_df, metab_df], axis=1)
    elif database == 'LR':
        adata.varm["database"] = LR_df
    else:
        adata.varm["database"] = metab_df

    return


def compute_knn_graph(
    adata: Union[str, anndata.AnnData],
    compute_neighbors_on_key: Optional[str] = None,
    distances_obsp_key: Optional[str] = None,
    weighted_graph: Optional[bool] = False,
    neighborhood_radius: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    neighborhood_factor: Optional[int] = 3,
    sample_key: Optional[str] = None,
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

    """
    start = time.time()

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

    # weights = make_weights_non_redundant(adata.obsp["weights"].toarray())
    # adata.obsp["weights"] = csr_matrix(weights)

    print("Finished computing the KNN graph in %.3f seconds" %(time.time()-start))

    return


def compute_local_autocorrelation(
    adata: AnnData,
    layer_key: Optional[Union[Literal["use_raw"], str]] = None,
    database_varm_key: Optional[str] = None,
    model: Optional[str] = None,
    genes: Optional[list] = None,
    use_metabolic_genes: bool = True,
    species: Optional[Union[Literal["mouse"], Literal["human"]]] = "mouse",
):

    start = time.time()
    print("Computing local autocorrelation...")

    adata.uns['layer_key'] = layer_key
    adata.uns['model'] = model

    if use_metabolic_genes and genes is None:
        genes = load_metabolic_genes(species)
        genes = adata.var_names[adata.var_names.isin(genes)]

    use_raw = layer_key == "use_raw"
    if (database_varm_key is not None) and (genes is None):
        metab_matrix = adata.varm[database_varm_key] if not use_raw else adata.raw.varm[database_varm_key]
        genes = metab_matrix.loc[(metab_matrix!=0).any(axis=1)].index
    elif (database_varm_key is None) and (genes is None):
        genes = adata.var_names if not use_raw else adata.raw.var.index

    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)

    weights = adata.obsp['weights']
    genes = genes[~np.all(counts == 0, axis=1)]
    counts = counts[~np.all(counts == 0, axis=1)]
    num_umi = np.array(counts.sum(axis=0))

    adata.uns['umi_counts'] = num_umi

    row_degrees = np.array(weights.sum(axis=1).T)[0]
    col_degrees = np.array(weights.sum(axis=0).T)[0]
    D = row_degrees + col_degrees

    Wtot2 = (weights.data ** 2).sum()

    def center_vals_f(x):
        return center_values_total(x, num_umi, model)
    counts = np.apply_along_axis(lambda x: center_vals_f(x)[np.newaxis], 1, counts).squeeze(axis=1)

    results = _compute_hs_inner_fast(counts.T, weights, Wtot2, D)
    results = pd.DataFrame(results, index=["G", "G_max", "EG", "stdG", "Z", "C"], columns=genes).T

    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]

    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"

    results = results[["C", "Z", "Pval", "FDR"]]

    adata.uns['gene_autocorrelation_results'] = results

    print("Finished computing local autocorrelation in %.3f seconds" %(time.time()-start))

    return


def load_metabolic_genes(
    species: Optional[Union[Literal["mouse"], Literal["human"]]] = None,
):

    metabolic_genes_paths = {
        'human': "/home/labs/nyosef/oier/Compass_data/metabolic_genes/metabolic_genes_h.csv",
        'mouse': "/home/labs/nyosef/oier/Compass_data/metabolic_genes/metabolic_genes_m.csv"
    }

    metabolic_genes = list(pd.read_csv(metabolic_genes_paths[species], index_col=0, header=None).index)

    return metabolic_genes


def compute_local_correlation(
    adata: AnnData,
    genes: Optional[list] = None,
):

    start = time.time()

    if genes is None:
        gene_autocorrelation_results = adata.uns['gene_autocorrelation_results']
        genes = gene_autocorrelation_results.loc[gene_autocorrelation_results.FDR < 0.05].sort_values('Z', ascending=False).index

    print(f"Computing pair-wise local correlation on {len(genes)} features...")
    
    layer_key = adata.uns['layer_key']
    model = adata.uns['model']

    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)

    weights = adata.obsp['weights']
    num_umi = counts.sum(axis=0)

    row_degrees = np.array(weights.sum(axis=1).T)[0]
    col_degrees = np.array(weights.sum(axis=0).T)[0]
    D = row_degrees + col_degrees

    counts = create_centered_counts(counts, model, num_umi)

    eg2s = ((weights @ counts.T) ** 2).sum(axis=0)

    results = _compute_hs_pairs_inner_centered_cond_sym_fast(counts, weights, eg2s)

    lcs, lc_zs = results

    lc_maxs = compute_local_cov_pairs_max(D, counts)
    lcs = lcs / lc_maxs

    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)

    adata.uns["lcs"] = lcs
    adata.uns["lc_zs"] = lc_zs

    print("Finished computing pair-wise local correlation in %.3f seconds" %(time.time()-start))

    return


def create_modules(
    adata: Union[str, anndata.AnnData],
    min_gene_threshold: Optional[int] = 15,
    fdr_threshold: Optional[float] = 0.05,
    z_threshold: Optional[float] = None,
    core_only: bool = False,
):
    """Assigns modules from the gene pair-wise Z-scores.

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
    start = time.time()
    print("Creating modules...")

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

    adata.uns["gene_modules"] = out_clusters
    adata.uns["gene_modules_dict"] = gene_modules_dict
    adata.uns["linkage"] = linkage_out

    print("Finished creating modules in %.3f seconds" %(time.time()-start))

    return


def calculate_module_scores(
    adata: AnnData,
    method: Optional[Union[Literal["PCA"], Literal["LDVAE"]]] = 'PCA',
):
    """Calculate Module Scores.

    In addition to returning its result, this method stores
    its output in the object at `self.module_scores`

    Returns
    -------
    module_scores: pandas.DataFrame
        Scores for each module for each gene
        Dimensions are genes x modules

    """
    start = time.time()

    layer_key = adata.uns['layer_key']
    model = adata.uns['model']

    use_raw = layer_key == "use_raw"
    modules = adata.uns["gene_modules_dict"]

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
                adata[:, module_genes],
                layer_key,
                model,
                umi_counts,
            )
        elif method == 'LDVAE':
            scores, loadings = compute_scores_LDVAE(
                adata[:, module_genes],
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

    print("Finished computing scores in %.3f seconds" %(time.time()-start))

    return


def integrate_vision_hotspot_results(
    adata: AnnData,
    cor_method: Optional[Union[Literal["pearson"], Literal["spearman"]]] = 'pearson',
):

    if ("vision_signatures" in adata.obsm) and (len(adata.uns["gene_modules_dict"].keys()) > 0):

        start = time.time()
        print("Integrating VISION and Hotspot results...")

        norm_data_key = adata.uns['norm_data_key']
        signature_varm_key = adata.uns['signature_varm_key']

        pvals_df, stats_df, FDR_df = compute_sig_mod_enrichment(adata, norm_data_key, signature_varm_key)
        adata.uns["sig_mod_enrichment_stats"] = stats_df
        adata.uns["sig_mod_enrichment_pvals"] = pvals_df
        adata.uns["sig_mod_enrichment_FDR"] = FDR_df

        if cor_method not in ['pearson', 'spearman']:
            raise ValueError(f'Invalid method: {cor_method}. Choose either "pearson" or "spearman".')

        cor_coef_df, cor_pval_df, cor_FDR_df = compute_sig_mod_correlation(adata, cor_method)
        adata.uns["sig_mod_correlation_coefs"] = cor_coef_df
        adata.uns["sig_mod_correlation_pvals"] = cor_pval_df
        adata.uns["sig_mod_correlation_FDR"] = cor_FDR_df

        adata.obsm["signature_modules_overlap"] = compute_signatures_anndata(
            adata,
            norm_data_key,
            signature_varm_key='signatures_overlap',
            signature_names_uns_key=None,
        )

        print("Finished integrating VISION and Hotspot results in %.3f seconds" %(time.time()-start))

    else:
        raise ValueError('Please make sure VISION has been run and Hotspot has identified at least one module.')

    return


def setup_deconv_adata(
    adata: Union[str, anndata.AnnData],
    compute_neighbors_on_key: Optional[str] = None,
    sample_key: Optional[str] = None,
    cell_type_list: Optional[list] = None,
    cell_type_key: Optional[str] = None,
    spot_diameter: Optional[int] = None,
):
    """Set up deconvolution AnnData.

    Parameters
    ----------
    adata
        AnnData object.
    compute_neighbors_on_key
        Key in `adata.obsm` to use for computing neighbors. If `None`, use neighbors stored in `adata`. If no neighbors have been previously computed an error will be raised.
    sample_key
        Sample information in case the data contains different samples or samples from different conditions. Input is key in `adata.obs`.
    cell_type_list
        Cell type or cluster information for the cell-cell communication analysis. Input is a list of keys in `adata.layers`.
    cell_type_key
        Cell type or cluster information for the cell-cell communication analysis. Input is key in `adata.obs`.

    """

    start = time.time()
    print("Setting up deconvolution data...")

    if isinstance(adata, str):
        adata = anndata.read(str)

    # setup AnnData
    deconv_adata = setup_anndata(
        adata,
        cell_type_list,
        compute_neighbors_on_key,
        cell_type_key,
        adata.uns['database_varm_key'],
        sample_key,
        spot_diameter,
        )
    deconv_adata.uns['cell_type_key'] = cell_type_key
    deconv_adata.uns['layer_key'] = None
    deconv_adata.uns['deconv_data'] = True

    if adata.uns["database"] in ['LR', 'both']:
        deconv_adata.uns['ligand'] = adata.uns['ligand']
        deconv_adata.uns['receptor'] = adata.uns['receptor']
        deconv_adata.uns['num_pairs'] = adata.uns['num_pairs']
        deconv_adata.uns['LR_database'] = adata.uns['LR_database']

    if adata.uns["database"] in ['transporter', 'both']:
        deconv_adata.uns['importer'] = adata.uns['importer']
        deconv_adata.uns['exporter'] = adata.uns['exporter']
        deconv_adata.uns['import_export'] = adata.uns['import_export']
        deconv_adata.uns['num_metabolites'] = adata.uns['num_metabolites']
        deconv_adata.uns['metabolite_database'] = adata.uns['metabolite_database']
    
    deconv_adata.uns["database"] = adata.uns["database"]

    print("Finished setting up deconvolution data in %.3f seconds" %(time.time()-start))

    return adata, deconv_adata


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


def compute_gene_pairs(
    adata: AnnData,
    layer_key: Optional[Union[Literal["use_raw"], str]] = None,
    cell_type_key: Optional[str] = None,
    cell_type_pairs: Optional[list] = None,
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
    elif 'cell_type_key' not in adata.uns and cell_type_key is None:
        raise ValueError('Please provide the "cell_type_key" argument.')

    use_raw = layer_key == "use_raw"
    genes = adata.raw.var.index if use_raw else adata.var_names

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

    if "filtered_genes_ct" not in adata.uns:
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

    if cell_type_pairs is None:
        cell_type_list = list(filtered_genes_ct.keys())
        cell_type_pairs = list(itertools.combinations(cell_type_list, 2))

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
    if "cell_type_pairs" not in adata.uns:
        adata.uns["cell_type_pairs"] = ct_pairs
    if "gene_pairs_per_metabolite" not in adata.uns:
        adata.uns["gene_pairs_per_metabolite"] = gene_pairs_per_metabolite
    if "gene_pairs_per_ct_pair" not in adata.uns:
        adata.uns["gene_pairs_per_ct_pair"] = gene_pairs_per_ct_pair
    
    print("Finished computing gene pairs in %.3f seconds" %(time.time()-start))

    return


def compute_cell_communication(
    adata: AnnData,
    layer_key: Optional[Union[Literal["use_raw"], str]] = None,
    model: Optional[str] = None,
    cell_type_key: Optional[str] = None,
    center_counts_for_np_test: Optional[bool] = False,
    M: Optional[int] = 1000,
    test: Optional[Union[Literal["parametric"], Literal["non-parametric"], Literal["both"]]] = None,
    njobs: Optional[int] = 1,
):
    
    start = time.time()
    print("Starting cell-cell communication analysis...")

    adata.uns['ccc_results'] = {}

    if test not in ['both', 'parametric', 'non-parametric']:
        raise ValueError('The "test" variable should be one of ["both", "parametric", "non-parametric"].')
    
    if 'cell_type_key' in adata.uns and cell_type_key is None:
        cell_type_key = adata.uns['cell_type_key']
    elif 'cell_type_key' not in adata.uns and cell_type_key is None:
        raise ValueError('Please provide the "cell_type_key" argument.')
    
    adata.uns['layer_key'] = layer_key
    adata.uns['model'] = model
    adata.uns['cell_type_key'] = cell_type_key
    adata.uns['center_counts_for_np_test'] = center_counts_for_np_test
    adata.uns['model'] = model

    with threadpool_limits(limits=njobs, user_api='blas'):
        run_cell_communication_analysis(adata, layer_key, model, cell_type_key, center_counts_for_np_test, M, test)

    print("Obtaining the communication results...")
    get_cell_communication_results(
        adata,
        adata.uns["genes"],
        layer_key,
        model,
        adata.uns["cell_types"],
        adata.uns["cell_type_pairs"],
        adata.uns["D"],
        test,
    )

    print("Finished computing cell-cell communication analysis in %.3f seconds" %(time.time()-start))

    return


def run_cell_communication_analysis(
    adata,
    layer_key,
    model,
    cell_type_key,
    center_counts_for_np_test,
    M,
    test,
):
    
    use_raw = layer_key == "use_raw"

    cells = adata.obs_names if not use_raw else adata.raw.obs.index
    cells = cells.values.astype(str)

    weights = sparse.COO.from_scipy_sparse(adata.obsp["weights"])
    cell_types = adata.obs[cell_type_key] if not use_raw else adata.raw.obs[cell_type_key]
    gene_pairs = adata.uns["gene_pairs"]
    genes = list(flatten(gene_pairs))
    adata.uns["genes"] = genes
    cell_type_pairs = adata.uns["cell_type_pairs"]
    gene_pairs_per_ct_pair = adata.uns["gene_pairs_per_ct_pair"]

    counts_1 = np.vstack([counts_from_anndata(adata[:, gene_pair[0]], layer_key, dense=True) if type(gene_pair[0]) is not list else gmean(counts_from_anndata(adata[:, gene_pair[0]], layer_key, dense=True), axis=0) for gene_pair in gene_pairs])
    counts_2 = np.vstack([counts_from_anndata(adata[:, gene_pair[1]], layer_key, dense=True) if type(gene_pair[1]) is not list else gmean(counts_from_anndata(adata[:, gene_pair[1]], layer_key, dense=True), axis=0) for gene_pair in gene_pairs])

    if test in ['parametric', 'both'] or center_counts_for_np_test:
        num_umi_1 = counts_1.sum(axis=0)
        counts_1_std = create_centered_counts(counts_1, model, num_umi_1)
        counts_1_std = np.nan_to_num(counts_1_std)

        num_umi_2 = counts_2.sum(axis=0)
        counts_2_std = create_centered_counts(counts_2, model, num_umi_2)
        counts_2_std = np.nan_to_num(counts_2_std)

    gene_pairs_ind = []
    for pair in gene_pairs:
        var1, var2 = pair
        var1_ind = genes.index(var1) if type(var1) is not list else [genes.index(var) for var in var1 if var in genes]
        var2_ind = genes.index(var2) if type(var2) is not list else [genes.index(var) for var in var2 if var in genes]
        pair_tuple = (var1_ind, var2_ind)
        gene_pairs_ind.append(pair_tuple)
    
    adata.uns["gene_pairs_ind"] = gene_pairs_ind

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

    weigths_ct_pairs = get_ct_pair_weights(
        weights, cell_type_pairs, cell_types
    )
    
    row_degrees = weigths_ct_pairs.sum(axis=2).todense()
    col_degrees = weigths_ct_pairs.sum(axis=1).todense()
    D = row_degrees + col_degrees
    adata.uns["D"] = D

    if test in ['parametric', 'both']:
        adata.uns['ccc_results']['p']={}
        adata.uns['ccc_results']['p']['cs'] = compute_CCC_scores(counts_1_std.T, counts_2_std.T, weigths_ct_pairs)

        weigths_ct_pairs_sq_data = weigths_ct_pairs.data ** 2
        weigths_ct_pairs_sq = sparse.COO(weigths_ct_pairs.coords, weigths_ct_pairs_sq_data, shape=weigths_ct_pairs.shape)
        Wtot2 = sparse.sum(weigths_ct_pairs_sq, axis=(1, 2))

        counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)
        num_umi = counts.sum(axis=0)
        counts = create_centered_counts(counts, model, num_umi)
        counts = np.nan_to_num(counts)
        counts = sparse.COO.from_numpy(counts)

        eg2s = conditional_eg2_cellcom(counts, weigths_ct_pairs)

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
        Z_pvals = norm.sf(Z_scores)

        adata.uns['ccc_results']['p']['stat'] = eg2s
        adata.uns['ccc_results']['p']['cs'] = C_scores
        adata.uns['ccc_results']['p']['Z'] = Z_scores
        adata.uns['ccc_results']['p']['Z_pval'] = Z_pvals
        adata.uns['ccc_results']['p']['Z_FDR'] = multipletests(Z_pvals.flatten(), method="fdr_bh")[1].reshape(Z_pvals.shape)

    if test in ["non-parametric", "both"]:
        adata.uns['ccc_results']['np']={}
        adata.uns['ccc_results']['np']['perm_cs'] = np.zeros((len(cell_type_pairs), counts_1.shape[0], M)).astype(np.float16)
        counts_1 = counts_1_std if center_counts_for_np_test else counts_1
        counts_2 = counts_2_std if center_counts_for_np_test else counts_2
        adata.uns['ccc_results']['np']['cs'] = compute_CCC_scores(counts_1.T, counts_2.T, weigths_ct_pairs) if not center_counts_for_np_test else adata.uns['ccc_results']['p']['cs']
        for i in tqdm(range(M)):
            idx = np.random.permutation(counts_1.shape[1])
            counts_1, counts_2 = counts_1[:, idx], counts_2[:, idx]
            cell_types_perm = pd.Series(cell_types[idx])

            weigths_ct_pairs_perm = get_ct_pair_weights(
                weights, cell_type_pairs, cell_types_perm
            )
            
            adata.uns['ccc_results']['np']['perm_cs'][:, :, i] = compute_CCC_scores(counts_1.T, counts_2.T, weigths_ct_pairs_perm)
        
        x = np.sum(adata.uns['ccc_results']['np']['perm_cs'] > adata.uns['ccc_results']['np']['cs'][:, :, np.newaxis], axis=2)
        pvals = (x + 1) / (M + 1)

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
        
        adata.uns['ccc_results']['np']['cs'] = C_scores
        adata.uns['ccc_results']['np']['pval'] = pvals
        adata.uns['ccc_results']['np']['FDR'] = multipletests(pvals.flatten(), method="fdr_bh")[1].reshape(pvals.shape)

        # counts_ct_pairs_t_null, counts_ct_pairs_u_null, weigths_ct_pairs_null = get_ct_pair_counts_and_weights_null(
        #     counts_ct_pairs_t, counts_ct_pairs_u, weights, cell_type_pairs, cell_types, M
        # )

    adata.uns["cell_types"] = cell_types.tolist()
    
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
    elif test == 'non-parametric':
        FDR_values = adata.uns['ccc_results']['cell_com_df']['FDR_np'].values
    else:
        raise ValueError('The "test" variable should be one of ["parametric", "non-parametric"].')

    adata.uns['ccc_results']['cell_com_df']['selected'] = (FDR_values < threshold)
    cell_com_df = adata.uns['ccc_results']['cell_com_df']
    adata.uns['ccc_results']['cell_com_df_sig'] = cell_com_df[cell_com_df.selected == True].copy()

    return


def compute_interacting_cell_scores(
    adata: Union[str, AnnData],
    func: Optional[Union[Literal["max"], Literal["min"], Literal["mean"], Literal["prod"]]] = 'mean',
    center_counts: Optional[bool] = False,
):
    
    # To visualize the results, the user should ask for a given ct pair and a metabolite/LR pair. More than one plot can also be visualized. Output an error if the asked ct pair/metabolite is not significant.

    model = adata.uns["model"]
    layer_key = adata.uns["layer_key"]
    use_raw = layer_key == "use_raw"

    cell_communication_df = adata.uns['ccc_results']['cell_com_df_sig']
    cell_communication_df[['Gene 1', 'Gene 2']] = cell_communication_df[['Gene 1', 'Gene 2']].map(lambda x: tuple(x) if isinstance(x, list) else x)

    gene_pairs_per_ct_pair = adata.uns['gene_pairs_per_ct_pair']
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
    results_dict = {k: v for d in results for k, v in d.items()}

    adata.uns['interacting_cell_scores'] = results_dict

    return 


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
    df_metabolites_grouped = pd.DataFrame(index=results.index)

    for metabolite, group in metabolite_gene_pair_df.groupby('metabolite'):
        gene_pairs = group['gene_pair'].tolist()
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
        bj = counts_u[gene_b_ind, j]
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


def compute_CCC_scores(
    counts_1: np.array,
    counts_2: np.array,
    weights: sparse.COO,
):

    scores = np.tensordot(weights, counts_1 * counts_2, axes=([2], [0])).sum(axis=1)

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


def conditional_eg2_cellcom(counts, weigths_ct_pairs):

    weigths_ct_pairs_sq_data = weigths_ct_pairs.data ** 2
    weigths_ct_pairs_sq = sparse.COO(weigths_ct_pairs.coords, weigths_ct_pairs_sq_data, shape=weigths_ct_pairs.shape)
    counts_sq = counts ** 2

    out_eg2 = np.tensordot(counts_sq, weigths_ct_pairs_sq, axes=([1], [1])).sum(axis=2).todense()

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


def get_cell_communication_results(
    adata, 
    genes,
    layer_key,
    model, 
    cell_types, 
    cell_type_pairs,
    D,
    test,
):
    
    gene_pairs_per_ct_pair = adata.uns['gene_pairs_per_ct_pair']
    gene_pairs_per_ct_pair_ind = adata.uns['gene_pairs_per_ct_pair_ind']

    counts = counts_from_anndata(adata[:, genes], layer_key, dense=True)

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

        c_values_norm = normalize_values(counts, cell_types, cell_type_pairs, gene_pairs_per_ct_pair_ind, c_values, D)
        adata.uns['ccc_results']['np']['cs_norm'] = c_values_norm
        cell_com_df['C_norm_np'] = c_values_norm.flatten()

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


def compute_max_cs(node_degrees, counts, gene_pairs_ind):

    # compute_max_cs_gp_partial = partial(
    #     compute_max_cs_gp,
    #     counts=counts.todense(),
    #     node_degrees=node_degrees,
    # )

    # result = np.array(list(map(compute_max_cs_gp_partial, gene_pairs_ind)))

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
