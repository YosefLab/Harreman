from itertools import zip_longest
from re import compile, match
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
import os
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

IMPORT_METAB_KEY = "IMPORT"
EXPORT_METAB_KEY = "EXPORT"
BOTH_METAB_KEY = "BOTH"


def extract_interaction_db(
    adata: AnnData,
    use_raw: bool = False,
    species: Optional[Union[Literal["mouse"], Literal["human"]]] = None,
    database: Optional[Union[Literal["transporter"], Literal["LR"], Literal["both"]]] = None,
    subset_genes: Optional[list] = None,
    subset_metabolites: Optional[list] = None,
    extracellular_only: Optional[bool] = True,
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
    database
        Whether to use the transporter database, the LR database, or both.

    Returns
    -------
    Genes by metabolites (or LRs) dataframe. Index is aligned to genes from adata.

    """

    if species not in ['human', 'mouse']:
        raise ValueError(f'species type: {species} is not supported currently. You should choose either "human" or "mouse".')

    if database is None:
        raise ValueError('Please one of the options to extract the interaction database: "transporter", "LR" or "both".')

    adata.uns['species'] = species
    
    if database == 'both' or database == 'LR':
        extract_lr_pairs(adata, species)
        index = adata.raw.var.index if use_raw else adata.var_names
        columns = adata.uns['LR_database'].index
        data = np.zeros((len(index), len(columns)))
        LR_df = pd.DataFrame(index=index, columns=columns, data=data)
        LR_df.index = LR_df.index.str.lower()
        for LR_name in columns:
            for key in ['ligand', 'receptor']:
                genes = adata.uns[key].loc[LR_name].dropna().values.tolist()
                if len(genes) > 0:
                    genes = pd.Index(genes).str.lower()
                    genes = genes.intersection(LR_df.index)
                    LR_df.loc[genes, LR_name] = 1.0 if key == 'ligand' else -1.0
        LR_df.index = index
        LR_df = LR_df.loc[:, (LR_df!=0).any(axis=0)]

    if database == 'both' or database == 'transporter':
        metab_dict = {}
        metab_dict = extract_transporter_info(adata, species, extracellular_only)
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
        database_df = pd.concat([LR_df, metab_df], axis=1)
    elif database == 'LR':
        database_df = LR_df
    else:
        database_df = metab_df
    
    if subset_genes:
        database_df.loc[~database_df.index.isin(subset_genes)] = 0
        database_df = database_df.loc[:, (database_df!=0).any(axis=0)]
    if subset_metabolites:
        database_df = database_df[subset_metabolites]

    adata.varm["database"] = database_df

    return


def extract_transporter_info(
    adata: AnnData,
    species: Optional[Union[Literal["mouse"], Literal["human"]]] = None,
    extracellular_only: bool = True,
    export_suffix: Tuple[str] = "(_exp|_export)",
    import_suffix: str = "(_imp|_import)",
    verbose: bool = False,
) -> Dict[str, Dict[str, List[str]]]:
    """Read csv file to extract the metabolite database."""

    BASE_PATH = '/home/labs/nyosef/oier/Harreman/data/HarremanDB'
    
    database_info = {
        'mouse': {
            'extracellular': os.path.join(BASE_PATH, 'HarremanDB_mouse_extracellular.csv'),
            'all': os.path.join(BASE_PATH, 'HarremanDB_mouse.csv'),
            'heterodimer': os.path.join(BASE_PATH, 'Heterodimer_info_mouse.csv'),
        },
        'human': {
            'extracellular': os.path.join(BASE_PATH, 'HarremanDB_human_extracellular.csv'),
            'all': os.path.join(BASE_PATH, 'HarremanDB_human.csv'),
            'heterodimer': os.path.join(BASE_PATH, 'Heterodimer_info_human.csv'),
        }
    }

    if extracellular_only:
        database_df = pd.read_csv(database_info[species]['extracellular'], index_col=0)
    else:
        database_df = pd.read_csv(database_info[species]['all'], index_col=0)
    
    heterodimer_info = pd.read_csv(database_info[species]['heterodimer'], index_col=0)

    directional_metabs = {}
    metab_genes_dir = {}
    directions = [IMPORT_METAB_KEY, EXPORT_METAB_KEY, BOTH_METAB_KEY]
    # Here \S is used as signature might have '-' in their name
    #  (\w is not suficient if number in signature for EX.)
    pattern_import = compile(r"(\S+)" + import_suffix + "$")
    pattern_export = compile(r"(\S+)" + export_suffix + "$")

    for i in range(len(database_df)):
        genes_split = database_df.loc[i, 'Gene'].split("/")
        genes_split = [gene.strip() for gene in genes_split] # Remove whitespaces just in case
        metabolite_full_name = database_df.loc[i, 'Metabolite']
        if len(genes_split) < 1:
            if verbose:
                print("Skipping empty entry; no genes for " + metabolite_full_name)
            continue
        # Skipping empty lines in the database
        if len(metabolite_full_name):
            z = match(pattern_import, metabolite_full_name.lower())
            if z:
                metabolite_name = z.groups()[0]
                direction = IMPORT_METAB_KEY
            else:
                z = match(pattern_export, metabolite_full_name.lower())
                if z:
                    metabolite_name = z.groups()[0]
                    direction = EXPORT_METAB_KEY
                else:
                    metabolite_name = metabolite_full_name
                    direction = BOTH_METAB_KEY
            # Get the gene names removing empty entry
            initial_value = genes_split
            gene_list = [x for x in initial_value if len(x)]
            # Double-check that the gene names are unique
            gene_list = list(np.unique(gene_list))
            if metabolite_name in directional_metabs.keys():
                directional_metabs[metabolite_name][direction] = gene_list
            else:
                directional_metabs[metabolite_name] = {direction: gene_list}
            if verbose:
                print(i, ": ", metabolite_full_name)

            gene_list_arr = pd.Series(gene_list).values[pd.Series(gene_list).isin(adata.var_names)]
            metab_genes_dir[metabolite_name] = {}
            for dir in directions:
                if dir == direction:
                    metab_genes_dir[metabolite_name][dir] = gene_list_arr
                else:
                    metab_genes_dir[metabolite_name][dir] = np.array([], dtype=object)

    metab_names = metab_genes_dir.keys()
    importer = [metab_genes_dir[metab][IMPORT_METAB_KEY] for metab in metab_names]
    exporter = [metab_genes_dir[metab][EXPORT_METAB_KEY] for metab in metab_names]
    import_export = [metab_genes_dir[metab][BOTH_METAB_KEY] for metab in metab_names]
    ind = database_df['Metabolite'].values
    adata.uns['importer'] = pd.DataFrame.from_records(zip_longest(*pd.Series(importer).values)).transpose()
    adata.uns['importer'].columns = ['Importer' + str(i) for i in range(adata.uns['importer'].shape[1])]
    adata.uns['importer'].index = ind
    adata.uns['exporter'] = pd.DataFrame.from_records(zip_longest(*pd.Series(exporter).values)).transpose()
    adata.uns['exporter'].columns = ['Exporter' + str(i) for i in range(adata.uns['exporter'].shape[1])]
    adata.uns['exporter'].index = ind
    adata.uns['import_export'] = pd.DataFrame.from_records(zip_longest(*pd.Series(import_export).values)).transpose()
    adata.uns['import_export'].columns = ['Importer_Exporter' + str(i) for i in range(adata.uns['import_export'].shape[1])]
    adata.uns['import_export'].index = ind
    adata.uns['num_metabolites'] = len(ind)
    adata.uns['metabolite_database'] = database_df
    adata.uns['heterodimer_info'] = heterodimer_info

    return directional_metabs


def extract_lr_pairs(adata, species):
    """Extracting LR pairs from CellChatDB."""

    BASE_PATH = '/home/labs/nyosef/oier/Harreman/data/CellChatDB'

    database_info = {
        'mouse': {
            'interaction': os.path.join(BASE_PATH, 'interaction_input_CellChatDB_v2_mouse.csv'),
            'complex': os.path.join(BASE_PATH, 'complex_input_CellChatDB_v2_mouse.csv'),
        },
        'human': {
            'interaction': os.path.join(BASE_PATH, 'interaction_input_CellChatDB_v2_human.csv'),
            'complex': os.path.join(BASE_PATH, 'complex_input_CellChatDB_v2_human.csv'),
        }
    }

    interaction = pd.read_csv(database_info[species]['interaction'], index_col=0)
    complex = pd.read_csv(database_info[species]['complex'], header=0, index_col=0)

    interaction = interaction.sort_values('annotation')
    ligand = interaction.ligand.values
    receptor = interaction.receptor.values
    interaction.pop('ligand')
    interaction.pop('receptor')

    for i in range(len(ligand)):
        for n in [ligand, receptor]:
            l = n[i]
            if l in complex.index:
                n[i] = complex.loc[l].dropna().values[pd.Series(complex.loc[l].dropna().values).isin(adata.var_names)]
            else:
                n[i] = pd.Series(l).values[pd.Series(l).isin(adata.var_names)]

    adata.uns['ligand'] = pd.DataFrame.from_records(zip_longest(*pd.Series(ligand).values)).transpose()
    adata.uns['ligand'].columns = ['Ligand' + str(i) for i in range(adata.uns['ligand'].shape[1])]
    adata.uns['ligand'].index = interaction.index
    adata.uns['receptor'] = pd.DataFrame.from_records(zip_longest(*pd.Series(receptor).values)).transpose()
    adata.uns['receptor'].columns = ['Receptor' + str(i) for i in range(adata.uns['receptor'].shape[1])]
    adata.uns['receptor'].index = interaction.index
    adata.uns['LR_database'] = interaction
    
    return


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
