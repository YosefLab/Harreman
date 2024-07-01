from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
import time
import anndata
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def counts_from_anndata(adata, layer_key, dense=False):

    if layer_key is None:
        counts = adata.X
    elif layer_key == "use_raw":
        counts = adata.raw.X
    else:
        counts = adata.layers[layer_key]
    counts = counts.transpose()

    is_sparse = issparse(counts)

    if not is_sparse:
        counts = np.asarray(counts)

    if dense:
        counts = counts.A if is_sparse else counts

    return counts


def setup_anndata(
        input_adata: anndata.AnnData,
        cell_types: list,
        compute_neighbors_on_key: str,
        cell_type_key: str,
        database_varm_key: str,
        sample_key: str,
        spot_diameter: int,
) -> anndata.AnnData:

    database = input_adata.varm[database_varm_key]
    barcode_key = 'barcodes'

    for i, ct in enumerate(cell_types):
        if ct not in input_adata.layers:
            continue
        adata = anndata.AnnData(input_adata.layers[ct])
        adata.obs_names = input_adata.obs_names.astype(str) + '_' + ct
        adata.var_names = input_adata.var_names
        adata.obsm[compute_neighbors_on_key] = input_adata.obsm[compute_neighbors_on_key]
        adata.obs[barcode_key] = input_adata.obs_names
        adata.obs[cell_type_key] = ct
        if sample_key is not None:
            adata.obs[sample_key] = input_adata.obs[sample_key].values

        if i == 0:
            out_adata = adata.copy()
        else:
            out_adata = anndata.concat([out_adata, adata])

    out_adata.uns['database_varm_key'] = input_adata.uns['database_varm_key']
    out_adata.uns['spot_diameter'] = spot_diameter
    out_adata.uns['barcode_key'] = barcode_key
    out_adata.varm[database_varm_key] = database

    valid_barcodes = out_adata.X.sum(axis=1) > 0
    out_adata = out_adata[valid_barcodes,].copy()

    return out_adata


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


# The code below belongs to the SpatialDM package. Modify it accordingly

def drop_uns_na(adata, global_stat=False, local_stat=False):
    adata.uns['geneInter'] = adata.uns['geneInter'].fillna('NA')
    adata.uns['global_res'] = adata.uns['global_res'].fillna('NA')
    adata.uns['ligand'] = adata.uns['ligand'].fillna('NA')
    adata.uns['receptor'] = adata.uns['receptor'].fillna('NA')
    adata.uns['local_stat']['n_spots'] = pd.DataFrame(adata.uns['local_stat']['n_spots'], columns=['n_spots'])
    if global_stat and ('global_stat' in adata.uns.keys()):
        adata.uns.pop('global_stat')
    if local_stat and ('local_stat' in adata.uns.keys()):
        adata.uns.pop('local_stat')

def restore_uns_na(adata):
    adata.uns['geneInter'] = adata.uns['geneInter'].replace('NA', np.nan)
    adata.uns['global_res'] = adata.uns['global_res'].replace('NA', np.nan)
    adata.uns['ligand'] = adata.uns['ligand'].replace('NA', np.nan)
    adata.uns['receptor'] = adata.uns['receptor'].replace('NA', np.nan)
    adata.uns['local_stat']['n_spots'] =  adata.uns['local_stat']['n_spots'].n_spots

def write_spatialdm_h5ad(adata, filename=None):
    if filename is None:
        filename = 'spatialdm_out.h5ad'
    elif not filename.endswith('h5ad'):
        filename = filename+'.h5ad'
    drop_uns_na(adata)
    adata.write(filename)

def read_spatialdm_h5ad(filename):
    adata = anndata.read_h5ad(filename)
    restore_uns_na(adata)
    return adata
