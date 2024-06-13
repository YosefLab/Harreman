   Genes by metabolites (or LRs) dataframe. Index is aligned to genes from adata.

    """
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

    if database == 'both':
        adata.varm["database"] = pd.concat([LR_df, metab_df], axis=1)
    elif database == 'LR':
        adata.varm["database"] = LR_df
    else:
        adata.varm["database"] = metab_df

    return


def extract_transporter_info(
    adata: AnnData,
    species: Optional[Union[Literal["mouse"], Literal["human"]]] = None,
    export_suffix: Tuple[str] = "(_exp|_export)",
    import_suffix: str = "(_imp|_import)",
    verbose: bool = False,
) -> Dict[str, Dict[str, List[str]]]:
    """Read csv file to extract the metabolite database."""
    if species == 'mouse':
        database_df = pd.read_csv('/home/labs/nyosef/oier/Cell_cell_communication/Metabolite_transporters/transporter_database_metabs_m.csv', index_col=0)
    elif species == 'human':
        database_df = pd.read_csv('/home/labs/nyosef/oier/Cell_cell_communication/Metabolite_transporters/transporter_database_metabs_h.csv', index_col=0)

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

    return directional_metabs


# Function obtained from SpatialDM (consider just importing the function and using it)
def extract_lr(adata, species, mean='algebra', min_cell=0, datahost='builtin'):
    """Find overlapping LRs from CellChatDB
    :param adata: AnnData object
    :param species: support 'human', 'mouse' and 'zebrafish'
    :param mean: 'algebra' (default) or 'geometric'
    :param min_cell: for each selected pair, the spots expressing ligand or receptor should be larger than the min,
    respectively.
    :param datahost: the host of the ligand-receptor data. 'builtin' for package built-in otherwise from figshare
    :return: ligand, receptor, geneInter (containing comprehensive info from CellChatDB) dataframes \
            in adata.uns.
    """
    if mean=='geometric':
        from scipy.stats.mstats import gmean
    adata.uns['mean'] = mean

    if datahost == 'package':
        if species in ['mouse', 'human', 'zerafish']:
            datapath = './datasets/LR_data/%s-' %(species)
        else:
            raise ValueError(f"species type: {species} is not supported currently. Please have a check.")

        import pkg_resources
        stream1 = pkg_resources.resource_stream(__name__, datapath + 'interaction_input_CellChatDB.csv.gz')
        geneInter = pd.read_csv(stream1, index_col=0, compression='gzip')

        stream2 = pkg_resources.resource_stream(__name__, datapath + 'complex_input_CellChatDB.csv')
        comp = pd.read_csv(stream2, header=0, index_col=0)
    else:
        if species == 'mouse':
            geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638919', index_col=0)
            comp = pd.read_csv('https://figshare.com/ndownloader/files/36638916', header=0, index_col=0)
        elif species == 'human':
            geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638943', header=0, index_col=0)
            comp = pd.read_csv('https://figshare.com/ndownloader/files/36638940', header=0, index_col=0)
        elif species == 'zebrafish':
            geneInter = pd.read_csv('https://figshare.com/ndownloader/files/38756022', header=0, index_col=0)
            comp = pd.read_csv('https://figshare.com/ndownloader/files/38756019', header=0, index_col=0)
        else:
            raise ValueError(f"species type: {species} is not supported currently. Please have a check.")

    geneInter = geneInter.sort_values('annotation')
    ligand = geneInter.ligand.values
    receptor = geneInter.receptor.values
    geneInter.pop('ligand')
    geneInter.pop('receptor')

    ## NOTE: the following for loop needs speed up
    # t = []
    for i in range(len(ligand)):
        for n in [ligand, receptor]:
            l = n[i]
            if l in comp.index:
                n[i] = comp.loc[l].dropna().values[pd.Series \
                    (comp.loc[l].dropna().values).isin(adata.var_names)]
            else:
                n[i] = pd.Series(l).values[pd.Series(l).isin(adata.var_names)]
        # if (len(ligand[i]) > 0) * (len(receptor[i]) > 0):
        #     if mean=='geometric':
        #         meanL = gmean(adata[:, ligand[i]].X, axis=1)
        #         meanR = gmean(adata[:, receptor[i]].X, axis=1)
        #     else:
        #         meanL = adata[:, ligand[i]].X.mean(axis=1)
        #         meanR = adata[:, receptor[i]].X.mean(axis=1)
        #     if (sum(meanL > 0) >= min_cell) * \
        #             (sum(meanR > 0) >= min_cell):
        #         t.append(True)
        #     else:
        #         t.append(False)
        # else:
        #     t.append(False)
    ind = geneInter.index
    adata.uns['ligand'] = pd.DataFrame.from_records(zip_longest(*pd.Series(ligand).values)).transpose()
    adata.uns['ligand'].columns = ['Ligand' + str(i) for i in range(adata.uns['ligand'].shape[1])]
    adata.uns['ligand'].index = ind
    adata.uns['receptor'] = pd.DataFrame.from_records(zip_longest(*pd.Series(receptor).values)).transpose()
    adata.uns['receptor'].columns = ['Receptor' + str(i) for i in range(adata.uns['receptor'].shape[1])]
    adata.uns['receptor'].index = ind
    adata.uns['num_pairs'] = len(ind)
    adata.uns['LR_database'] = geneInter.loc[ind]
    if adata.uns['num_pairs'] == 0:
        raise ValueError("No effective RL. Please have a check on input count matrix/species.")
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
