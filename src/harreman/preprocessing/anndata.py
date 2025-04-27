from typing import Optional, Union
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


# The code below has been adapted from the SpatialDM package.

def modify_uns_hotspot(adata):
    if 'modules' in adata.uns.keys():
        adata.var['modules'] = adata.uns['modules']
        del adata.uns['modules']
    
    if 'super_modules' in adata.uns.keys():
        adata.var['super_modules'] = adata.uns['super_modules']
        del adata.uns['super_modules']
    
    if 'lc_zs' in adata.uns.keys():
        genes = [' - '.join(gene) if isinstance(gene, tuple) else gene for gene in adata.uns['lc_zs'].columns]
        adata.uns['lc_zs'].index = genes
        adata.uns['lc_zs'].columns = genes
    
    return


def modify_uns_harreman(adata):
    uns_keys = ['ligand', 'receptor', 'LR_database', 'import_export']
    for uns_key in uns_keys:
        if uns_key in adata.uns.keys():
            adata.uns[uns_key] = adata.uns[uns_key].fillna('NA')
    
    if 'LR_database' in adata.uns.keys():
        adata.uns['LR_database'].columns = [col.replace('.', '_') for col in adata.uns['LR_database'].columns]
        adata.uns['LR_database']['ligand_transmembrane'] = adata.uns['LR_database']['ligand_transmembrane'].astype(str)
        adata.uns['LR_database']['receptor_transmembrane'] = adata.uns['LR_database']['receptor_transmembrane'].astype(str)

    if 'gene_pairs' in adata.uns.keys():
        gene_pairs_tmp = [(x, ' - '.join(y) if isinstance(y, list) else y) for x, y in adata.uns['gene_pairs']]
        gene_pairs_tmp = [(' - '.join(x) if isinstance(x, list) else x, y) for x, y in gene_pairs_tmp]
        adata.uns['gene_pairs'] = ['_'.join(gp) for gp in gene_pairs_tmp]

    if 'gene_pairs_ind' in adata.uns.keys():
        gene_pairs_ind_tmp = [(x, ' - '.join(map(str, y)) if isinstance(y, list) else str(y)) for x, y in adata.uns['gene_pairs_ind']]
        gene_pairs_ind_tmp = [(' - '.join(map(str, x)) if isinstance(x, list) else str(x), y) for x, y in gene_pairs_ind_tmp]
        adata.uns['gene_pairs_ind'] = ['_'.join(gp) for gp in gene_pairs_ind_tmp]
    
    if 'gene_pairs_per_metabolite' in adata.uns.keys():
        adata.uns['gene_pairs_per_metabolite'] = {key: {
            'gene_pair': [(' - '.join(gp_1) if isinstance(gp_1, list) else gp_1, gp_2) for gp_1, gp_2 in subdict['gene_pair']],
            'gene_type': subdict['gene_type']
        } for key, subdict in adata.uns['gene_pairs_per_metabolite'].items()}
        adata.uns['gene_pairs_per_metabolite'] = {key: {
            'gene_pair': [(gp_1, ' - '.join(gp_2) if isinstance(gp_2, list) else gp_2) for gp_1, gp_2 in subdict['gene_pair']],
            'gene_type': subdict['gene_type']
        } for key, subdict in adata.uns['gene_pairs_per_metabolite'].items()}
    
    if 'gene_pairs_per_ct_pair' in adata.uns.keys():
        adata.uns['gene_pairs_per_ct_pair'] = {key: [(x, ' - '.join(y) if isinstance(y, list) else y) for x, y in tuples_list] for key, tuples_list in adata.uns['gene_pairs_per_ct_pair'].items()}
        adata.uns['gene_pairs_per_ct_pair'] = {key: [(' - '.join(x) if isinstance(x, list) else x, y) for x, y in tuples_list] for key, tuples_list in adata.uns['gene_pairs_per_ct_pair'].items()}
        adata.uns['gene_pairs_per_ct_pair'] = {' - '.join(key): value for key, value in adata.uns['gene_pairs_per_ct_pair'].items()}
    if 'gene_pairs_per_ct_pair_ind' in adata.uns.keys():
        adata.uns['gene_pairs_per_ct_pair_ind'] = {' - '.join(key): value for key, value in adata.uns['gene_pairs_per_ct_pair_ind'].items()}
    if 'gene_pairs_ind_per_ct_pair' in adata.uns.keys():
        adata.uns['gene_pairs_ind_per_ct_pair'] = {key: [(x, ' - '.join(map(str, y)) if isinstance(y, list) else y) for x, y in tuples_list] for key, tuples_list in adata.uns['gene_pairs_ind_per_ct_pair'].items()}
        adata.uns['gene_pairs_ind_per_ct_pair'] = {key: [(' - '.join(map(str, x)) if isinstance(x, list) else x, y) for x, y in tuples_list] for key, tuples_list in adata.uns['gene_pairs_ind_per_ct_pair'].items()}
        adata.uns['gene_pairs_ind_per_ct_pair'] = {' - '.join(key): value for key, value in adata.uns['gene_pairs_ind_per_ct_pair'].items()}

    if 'ccc_results' in adata.uns.keys():
        adata.uns['ccc_results']['cell_com_df_gp'] = adata.uns['ccc_results']['cell_com_df_gp'].applymap(lambda x: ' - '.join(x) if isinstance(x, list) else x)
        adata.uns['ccc_results']['cell_com_df_m'] = adata.uns['ccc_results']['cell_com_df_m'].applymap(lambda x: ' - '.join(x) if isinstance(x, list) else x)
        if 'cell_com_df_gp_sig' in adata.uns['ccc_results'].keys():
            adata.uns['ccc_results']['cell_com_df_gp_sig'] = adata.uns['ccc_results']['cell_com_df_gp_sig'].applymap(lambda x: ' - '.join(x) if isinstance(x, list) else x)
            adata.uns['ccc_results']['cell_com_df_m_sig'] = adata.uns['ccc_results']['cell_com_df_m_sig'].applymap(lambda x: ' - '.join(x) if isinstance(x, list) else x)
        if 'cell_com_df_sig_metab' in adata.uns['ccc_results'].keys():
            adata.uns['ccc_results']['cell_com_df_sig_metab']['Gene 1'] = [list(i) if isinstance(i, tuple) else i for i in adata.uns['ccc_results']['cell_com_df_sig_metab']['Gene 1'].values]
            adata.uns['ccc_results']['cell_com_df_sig_metab']['Gene 2'] = [list(i) if isinstance(i, tuple) else i for i in adata.uns['ccc_results']['cell_com_df_sig_metab']['Gene 2'].values]
            adata.uns['ccc_results']['cell_com_df_sig_metab'] = adata.uns['ccc_results']['cell_com_df_sig_metab'].applymap(lambda x: ' - '.join(x) if isinstance(x, list) else x)
            if adata.uns['ccc_results']['cell_com_df_sig_metab'].shape[0] > 0:
                adata.uns['ccc_results']['cell_com_df_sig_metab'][['gene_type1', 'gene_type2']] = pd.DataFrame(adata.uns['ccc_results']['cell_com_df_sig_metab']['gene_type'].tolist(), index=adata.uns['ccc_results']['cell_com_df_sig_metab'].index)
            adata.uns['ccc_results']['cell_com_df_sig_metab'] = adata.uns['ccc_results']['cell_com_df_sig_metab'].drop(['gene_pair', 'gene_type'] ,axis=1)

    if 'metabolite_gene_pair_df' in adata.uns.keys():
        adata.uns['metabolite_gene_pair_df'][['gene_pair1', 'gene_pair2']] = pd.DataFrame(adata.uns['metabolite_gene_pair_df']['gene_pair'].tolist(), index=adata.uns['metabolite_gene_pair_df'].index)
        adata.uns['metabolite_gene_pair_df'][['gene_type1', 'gene_type2']] = pd.DataFrame(adata.uns['metabolite_gene_pair_df']['gene_type'].tolist(), index=adata.uns['metabolite_gene_pair_df'].index)
        adata.uns['metabolite_gene_pair_df'] = adata.uns['metabolite_gene_pair_df'].drop(['gene_pair', 'gene_type'] ,axis=1)
        adata.uns['metabolite_gene_pair_df']['gene_pair1'] = [list(i) if isinstance(i, tuple) else i for i in adata.uns['metabolite_gene_pair_df']['gene_pair1'].values]
        adata.uns['metabolite_gene_pair_df']['gene_pair2'] = [list(i) if isinstance(i, tuple) else i for i in adata.uns['metabolite_gene_pair_df']['gene_pair2'].values]
        adata.uns['metabolite_gene_pair_df'] = adata.uns['metabolite_gene_pair_df'].applymap(lambda x: ' - '.join(x) if isinstance(x, list) else x)

    return


def write_h5ad(
    adata: anndata.AnnData, 
    filename: Optional[str] = None,
):
    if filename is None:
        raise ValueError('Please provide the path to save the file.')
    elif not filename.endswith('h5ad'):
        filename = filename+'.h5ad'
    
    if 'distances' in adata.obsp.keys():
        adata.obsp['distances'] = adata.obsp['distances'].tocsr()
    
    modify_uns_hotspot(adata)
    modify_uns_harreman(adata)
    adata.write(filename)


def read_h5ad(filename):
    adata = anndata.read_h5ad(filename)

    if 'genes' in adata.uns.keys():
        adata.uns['genes'] = list(adata.uns['genes'])
    
    if 'gene_pairs' in adata.uns.keys():
        gene_pairs_tmp = [tuple(gp.split('_')) for gp in adata.uns['gene_pairs']]
        gene_pairs_tmp = [(x, list(y.split(' - ')) if ' - ' in y else y) for x, y in gene_pairs_tmp]
        adata.uns['gene_pairs'] = [(list(x.split(' - ')) if ' - ' in x else x, y) for x, y in gene_pairs_tmp]
        
    if 'gene_pairs_ind' in adata.uns.keys():
        gene_pairs_ind_tmp = [tuple(gp.split('_')) for gp in adata.uns['gene_pairs_ind']]
        gene_pairs_ind_tmp = [(x, list(int(val) for val in y.split(' - ')) if ' - ' in y else int(y)) for x, y in gene_pairs_ind_tmp]
        adata.uns['gene_pairs_ind'] = [(list(int(val) for val in x.split(' - ')) if ' - ' in x else int(x), y) for x, y in gene_pairs_ind_tmp]
    
    uns_keys = ['ligand', 'receptor', 'LR_database', 'import_export']
    for uns_key in uns_keys:
        if uns_key in adata.uns.keys():
            adata.uns[uns_key] = adata.uns[uns_key].replace("NA", np.nan)

    return adata
