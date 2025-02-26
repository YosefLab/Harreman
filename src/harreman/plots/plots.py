import warnings
from typing import Optional, Union, Literal
from anndata import AnnData
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import leaves_list


def local_correlation_plot(
    adata: AnnData,
    mod_cmap='tab10',
    vmin=-10,
    vmax=10,
    z_cmap='RdBu_r',
    yticklabels=False,
    use_super_modules=False,
    show=True,
):
    
    local_correlation_z = adata.uns["lc_zs"]
    modules = adata.uns["super_modules"] if use_super_modules else adata.uns["modules"]
    linkage = adata.uns["linkage"]

    row_colors = None
    colors = list(plt.get_cmap(mod_cmap).colors)
    module_colors = {i: colors[(i-1) % len(colors)] for i in modules.unique()}
    module_colors[-1] = '#ffffff'

    row_colors1 = pd.Series(
        [module_colors[i] for i in modules],
        index=local_correlation_z.index,
    )

    row_colors = pd.DataFrame({
        "Modules": row_colors1,
    })

    cm = sns.clustermap(
        local_correlation_z,
        row_linkage=linkage,
        col_linkage=linkage,
        vmin=vmin,
        vmax=vmax,
        cmap=z_cmap,
        xticklabels=False,
        yticklabels=yticklabels,
        row_colors=row_colors,
        rasterized=True,
    )

    fig = plt.gcf()
    plt.sca(cm.ax_heatmap)
    plt.ylabel("")
    plt.xlabel("")

    cm.ax_row_dendrogram.remove()

    # Add 'module X' annotations
    ii = leaves_list(linkage)

    mod_reordered = modules.iloc[ii]
    
    adata.uns['mod_reordered'] = [mod for mod in mod_reordered.unique() if mod != -1]

    mod_map = {}
    y = np.arange(modules.size)

    for x in mod_reordered.unique():
        if x == -1:
            continue

        mod_map[x] = y[mod_reordered == x].mean()

    plt.sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        plt.text(-.5, y=mod_y, s="Module {}".format(mod),
                 horizontalalignment='right',
                 verticalalignment='center')
    plt.xticks([])

    # Find the colorbar 'child' and modify
    min_delta = 1e99
    min_aa = None
    for aa in fig.get_children():
        try:
            bbox = aa.get_position()
            delta = (0-bbox.xmin)**2 + (1-bbox.ymax)**2
            if delta < min_delta:
                delta = min_delta
                min_aa = aa
        except AttributeError:
            pass

    min_aa.set_ylabel('Z-Scores')
    min_aa.yaxis.set_label_position("left")
    
    if show:
        plt.show()


def average_local_correlation_plot(
    adata: AnnData,
    mod_cmap='tab10',
    vmin=-10,
    vmax=10,
    cor_cmap='RdBu_r',
    yticklabels=False,
    row_cluster=True,
    col_cluster=True,
    use_super_modules=False,
    show=True,
):
    
    local_correlation_z = adata.uns["lc_zs"]
    modules = adata.uns["super_modules"] if use_super_modules else adata.uns["modules"]
    
    avg_local_correlation_z = local_correlation_z.copy()
    avg_local_correlation_z['module_row'] = modules
    avg_local_correlation_z = avg_local_correlation_z.set_index('module_row', append=True)
    avg_local_correlation_z.columns = pd.MultiIndex.from_arrays([modules[avg_local_correlation_z.columns].values, avg_local_correlation_z.columns])

    avg_local_correlation_z = avg_local_correlation_z.groupby(level=1).mean().groupby(level=0, axis=1).mean()
    avg_local_correlation_z = avg_local_correlation_z.loc[avg_local_correlation_z.index != -1, avg_local_correlation_z.columns != -1]
    mod_reordered = adata.uns['mod_reordered']
    avg_local_correlation_z = avg_local_correlation_z.loc[mod_reordered, mod_reordered]

    row_colors = None
    colors = list(plt.get_cmap(mod_cmap).colors)
    module_colors = {i: colors[(i-1) % len(colors)] for i in modules.unique()}
    module_colors[-1] = '#ffffff'

    row_colors = pd.DataFrame({
        "Modules": module_colors,
    })

    cm = sns.clustermap(
        avg_local_correlation_z,
        vmin=vmin,
        vmax=vmax,
        cmap=cor_cmap,
        xticklabels=False,
        yticklabels=yticklabels,
        row_colors=row_colors,
        rasterized=True,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
    )

    fig = plt.gcf()
    plt.sca(cm.ax_heatmap)
    plt.ylabel("")
    plt.xlabel("")

    cm.ax_row_dendrogram.remove()

    if row_cluster:
        reordered_indices = cm.dendrogram_row.reordered_ind
        mod_reordered = [avg_local_correlation_z.index[i] for i in reordered_indices]

    mod_map = {}
    y = np.arange(len(mod_reordered))

    for x in mod_reordered:
        if x == -1:
            continue

        mod_map[x] = y[mod_reordered == x].mean() + 0.5

    plt.sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        plt.text(-.5, y=mod_y, s="Module {}".format(mod),
                    horizontalalignment='right',
                    verticalalignment='center')
    plt.xticks([])

    # Find the colorbar 'child' and modify
    min_delta = 1e99
    min_aa = None
    for aa in fig.get_children():
        try:
            bbox = aa.get_position()
            delta = (0-bbox.xmin)**2 + (1-bbox.ymax)**2
            if delta < min_delta:
                delta = min_delta
                min_aa = aa
        except AttributeError:
            pass

    min_aa.set_ylabel('Avg. local correlation Z')
    min_aa.yaxis.set_label_position("left")

    if show:
        plt.show()


def module_score_correlation_plot(
    adata: AnnData,
    mod_cmap='tab10',
    vmin=-1,
    vmax=1,
    cor_cmap='RdBu_r',
    yticklabels=False,
    method='pearson',
    use_super_modules=False,
    row_cluster=True,
    col_cluster=True,
    show=True,
):
    
    module_scores = adata.obsm['super_module_scores'] if use_super_modules else adata.obsm['module_scores']
    modules = adata.uns["super_modules"] if use_super_modules else adata.uns["modules"]

    cor_matrix = module_scores.corr(method)
    mod_int = [int(mod.split(' ')[1]) for mod in cor_matrix.index]
    cor_matrix.index = cor_matrix.columns = mod_int
    mod_reordered = adata.uns['mod_reordered']
    cor_matrix = cor_matrix.loc[mod_reordered, mod_reordered]

    row_colors = None
    colors = list(plt.get_cmap(mod_cmap).colors)
    module_colors = {i: colors[(i-1) % len(colors)] for i in modules.unique()}
    module_colors[-1] = '#ffffff'

    row_colors = pd.DataFrame({
        "Modules": module_colors,
    })

    cm = sns.clustermap(
        cor_matrix,
        vmin=vmin,
        vmax=vmax,
        cmap=cor_cmap,
        xticklabels=False,
        yticklabels=yticklabels,
        row_colors=row_colors,
        rasterized=True,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
    )

    fig = plt.gcf()
    plt.sca(cm.ax_heatmap)
    plt.ylabel("")
    plt.xlabel("")

    cm.ax_row_dendrogram.remove()

    if row_cluster:
        reordered_indices = cm.dendrogram_row.reordered_ind
        mod_reordered = [cor_matrix.index[i] for i in reordered_indices]

    mod_map = {}
    y = np.arange(len(mod_reordered))

    for x in mod_reordered:
        if x == -1:
            continue

        mod_map[x] = y[mod_reordered == x].mean() + 0.5

    plt.sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        plt.text(-.5, y=mod_y, s="Module {}".format(mod),
                    horizontalalignment='right',
                    verticalalignment='center')
    plt.xticks([])

    # Find the colorbar 'child' and modify
    min_delta = 1e99
    min_aa = None
    for aa in fig.get_children():
        try:
            bbox = aa.get_position()
            delta = (0-bbox.xmin)**2 + (1-bbox.ymax)**2
            if delta < min_delta:
                delta = min_delta
                min_aa = aa
        except AttributeError:
            pass

    min_aa.set_ylabel(f'{method.capitalize()} R')
    min_aa.yaxis.set_label_position("left")
    
    if show:
        plt.show()


def plot_interacting_cell_scores(
    adata: AnnData,
    deconv_adata: Optional[AnnData] = None,
    cell_type_pair: Optional[list] = None,
    interactions: Optional[list] = None,
    coords_obsm_key: Optional[str] = None,
    test: Optional[Union[Literal["parametric"], Literal["non-parametric"]]] = None,
    only_sig_values: Optional[bool] = False,
    use_FDR: Optional[bool] = True,
    normalize_values: Optional[bool] = False,
    sample_specific: Optional[bool] = False,
    s: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[tuple] = (10,10),
    cmap: Optional[str] = 'Reds',
    colorbar: Optional[bool] = True,
    swap_y_axis: Optional[bool] = False,
):
    
    if isinstance(vmin, str) and 'p' not in vmin:
        raise ValueError('"vmin" needs to be either a numeric value or a percentile: e.g. "p5".')
    if isinstance(vmax, str) and 'p' not in vmax:
        raise ValueError('"vmax" needs to be either a numeric value or a percentile: e.g. "p95".')
    
    if test not in ['parametric', 'non-parametric']:
        raise ValueError('The "test" variable should be one of ["parametric", "non-parametric"].')
    
    test_str = 'p' if test == 'parametric' else 'np'
    
    if sample_specific and 'sample_key' not in adata.uns.keys():
        raise ValueError('Sample information not found. Run Harreman using the "sample_key" parameter.')
    
    if deconv_adata is not None:
        adata.uns['interacting_cell_results'] = deconv_adata.uns['interacting_cell_results']
    
    if only_sig_values:
        sig_str = 'FDR' if use_FDR else 'pval'
        interacting_cell_scores_gp = adata.uns['interacting_cell_results'][test_str]['gp'][f'cs_sig_{sig_str}']
        interacting_cell_scores_m = adata.uns['interacting_cell_results'][test_str]['m'][f'cs_sig_{sig_str}']
    else:
        interacting_cell_scores_gp = adata.uns['interacting_cell_results'][test_str]['gp']['cs']
        interacting_cell_scores_m = adata.uns['interacting_cell_results'][test_str]['m']['cs']
    
    if interactions is None:
        raise ValueError("Please provide a LR pair or a metabolite.")
    
    interacting_cell_scores_gp = pd.DataFrame(interacting_cell_scores_gp, index=adata.obs_names, columns=adata.uns['gene_pairs_sig_names'])
    interacting_cell_scores_m = pd.DataFrame(interacting_cell_scores_m, index=adata.obs_names, columns=adata.uns['metabolites'])
    gene_pairs = [inter for inter in interactions if inter in adata.uns['gene_pairs_sig_names']]
    metabs = [inter for inter in interactions if inter in adata.uns['metabolites']]
    if len(gene_pairs) > 0 and len(metabs) > 0:
        interacting_cell_scores = pd.concat([interacting_cell_scores_gp, interacting_cell_scores_m], axis=1)
    elif len(gene_pairs) == 0 and len(metabs) == 0:
        raise ValueError("The provided LR pairs and/or metabolites don't have significant interactions.")
    else:
        interacting_cell_scores = interacting_cell_scores_gp if len(gene_pairs) > 0 else interacting_cell_scores_m
    
    if isinstance(cell_type_pair, list):
        if len(cell_type_pair) != 2:
            raise ValueError("Please provide two cell types.")
        else:
            ct_1, ct_2 = cell_type_pair
        
        if (ct_1, ct_2) in interacting_cell_scores.keys():
            ct_pair = (ct_1, ct_2)
        elif (ct_2, ct_1) in interacting_cell_scores.keys():
            ct_pair = (ct_2, ct_1)
        else:
            raise ValueError(f"Cell types {ct_1} and {ct_2} don't have significant interactions. Please provide another cell type pair.")
    else:
        ct_pair = cell_type_pair
    
    if ct_pair is None:
        if isinstance(interacting_cell_scores, pd.DataFrame):
            scores = interacting_cell_scores[interactions]
        else:
            if 'combined_cell_scores' not in adata.uns:
                sum_ct_pair_scores(adata)
            scores = adata.uns['combined_cell_scores'][interactions]
    else:
        scores = interacting_cell_scores[ct_pair][interactions]
    
    if normalize_values:
        scores = scores.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0) #We apply min-max normalization
    
    for interaction in interactions:
        if isinstance(vmin, str):
            vmin_new = int(vmin.split('p')[1])
            vmin_new = np.percentile(scores[interaction], vmin_new)
        else:
            vmin_new = vmin
        if isinstance(vmax, str):
            vmax_new = int(vmax.split('p')[1])
            vmax_new = np.percentile(scores[interaction], vmax_new)
        else:
            vmax_new = vmax
        
        if sample_specific:
            sample_key = adata.uns['sample_key']
            for sample in adata.obs[sample_key].unique().tolist():
                print(sample)
                plot_interaction(adata[adata.obs[sample_key] == sample], scores.loc[adata.obs[sample_key] == sample], interaction, ct_pair, coords_obsm_key, s, vmin_new, vmax_new, figsize, cmap, colorbar, swap_y_axis)
                plt.show()
                plt.close()
        else:
            plot_interaction(adata, scores, interaction, ct_pair, coords_obsm_key, s, vmin_new, vmax_new, figsize, cmap, colorbar, swap_y_axis)
            plt.show()
            plt.close()


def sum_ct_pair_scores(adata):

    interacting_cell_scores = adata.uns['interacting_cell_scores']

    df_list = []
    for key in interacting_cell_scores.keys():
        df = interacting_cell_scores[key].copy()
        df_list.append(df)
    
    combined_df = pd.concat(df_list, axis=1)
    combined_scores = combined_df.T.groupby(level=0).sum().T

    adata.uns['combined_cell_scores'] = combined_scores

    return


def plot_interaction(adata, scores, interaction, ct_pair, coords_obsm_key, s, vmin, vmax, figsize, cmap, colorbar, swap_y_axis):

    if isinstance(adata.obsm[coords_obsm_key], pd.DataFrame):
        coords = adata.obsm[coords_obsm_key].values
    else:
        coords = adata.obsm[coords_obsm_key]
    
    # plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    ax.set_aspect('equal', adjustable='box')
    _prettify_axis(ax, spatial=True)
    if swap_y_axis:
        plt.scatter(coords[:,0], -coords[:,1], c=scores[interaction], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
    else:
        plt.scatter(coords[:,0], coords[:,1], c=scores[interaction], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
    plt.title(f'{interaction}; {ct_pair[0]} and {ct_pair[1]}') if ct_pair is not None else plt.title(f'{interaction}')
    if colorbar:
        plt.colorbar()


def plot_NMF_results(
    adata: AnnData,
    deconv_adata: Optional[AnnData] = None,
    coords_obsm_key: Optional[str] = None,
    interaction_type: Optional[Union[Literal["metabolite"], Literal["gene_pair"]]] = "metabolite",
    only_sig_values: Optional[bool] = False,
    use_FDR: Optional[bool] = True,
    n_factors: Optional[int] = 5,
    s: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[tuple] = (10,10),
    cmap: Optional[str] = 'Reds',
    colorbar: Optional[bool] = True,
    swap_y_axis: Optional[bool] = False,
):
    
    if isinstance(vmin, str) and 'p' not in vmin:
        raise ValueError('"vmin" needs to be either a numeric value or a percentile: e.g. "p5".')
    if isinstance(vmax, str) and 'p' not in vmax:
        raise ValueError('"vmax" needs to be either a numeric value or a percentile: e.g. "p95".')
    
    if deconv_adata is not None:
        adata.uns['NMF_results'] = deconv_adata.uns['NMF_results']
    
    interaction_type_str = 'm' if interaction_type == 'metabolite' else 'gp'

    if only_sig_values:
        sig_str = 'FDR' if use_FDR else 'pval'
    
    NMF_results_key_W = f'NMF_W_{interaction_type_str}_{n_factors}_sig_{sig_str}' if only_sig_values else f'NMF_W_{interaction_type_str}_{n_factors}'
    if NMF_results_key_W not in adata.uns["NMF_results"].keys():
        raise ValueError("The provided parameters haven't been used to compute NMF. Input the correct parameters.")
    
    W = adata.uns["NMF_results"][NMF_results_key_W]
    factors = W.columns
    
    for factor in factors:
        if isinstance(vmin, str):
            vmin_new = int(vmin.split('p')[1])
            vmin_new = np.percentile(W[factor], vmin_new)
        else:
            vmin_new = vmin
        if isinstance(vmax, str):
            vmax_new = int(vmax.split('p')[1])
            vmax_new = np.percentile(W[factor], vmax_new)
        else:
            vmax_new = vmax
        
        plot_factor(adata, W, factor, coords_obsm_key, s, vmin_new, vmax_new, figsize, cmap, colorbar, swap_y_axis)
        plt.show()
        plt.close()


def plot_factor(adata, W, factor, coords_obsm_key, s, vmin, vmax, figsize, cmap, colorbar, swap_y_axis):

    if isinstance(adata.obsm[coords_obsm_key], pd.DataFrame):
        coords = adata.obsm[coords_obsm_key].values
    else:
        coords = adata.obsm[coords_obsm_key]
    
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    _prettify_axis(ax, spatial=True)
    if swap_y_axis:
        plt.scatter(coords[:,0], -coords[:,1], c=W[factor], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
    else:
        plt.scatter(coords[:,0], coords[:,1], c=W[factor], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
    plt.title(factor)
    if colorbar:
        plt.colorbar()


def _prettify_axis(ax, spatial=False):
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if spatial:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Spatial1")
        plt.ylabel("Spatial2")


def plot_signature_for_selection(adata, signature, coords_obsm_key, s, vmin, vmax, figsize, cmap, colorbar):

    scores = adata.obsm['vision_signatures']

    if isinstance(adata.obsm[coords_obsm_key], pd.DataFrame):
        coords = adata.obsm[coords_obsm_key].values
    else:
        coords = adata.obsm[coords_obsm_key]
    
    points = np.column_stack([coords[:,0], coords[:,1]])
    
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    _prettify_axis(ax, spatial=True)
    p = plt.scatter(coords[:,0], coords[:,1], c=scores[signature], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
    plt.title(signature)
    if colorbar:
        plt.colorbar()
    
    return p, ax, points


def plot_selection_histplot(adata, signature, group):

    adata.obs['selected'] = group
    adata.obs['selected'][adata.obs['selected'] == 1] = 'Selection'
    adata.obs['selected'][adata.obs['selected'] == 0] = 'Remainder'

    if signature not in adata.obs:
        adata.obs[signature] = adata.obsm['vision_signatures'][signature]

    ax = sns.histplot(data=adata.obs, x=signature, hue=adata.obs['selected'].tolist(), bins=30, palette={'Selection': '#FF7F00', 'Remainder': '#1F78B4'})
    plt.show()

    return


def plot_vision_autocorrelation(
    adata, 
    type: Optional[Union[Literal["observations"], Literal["signatures"]]] = None,
    center: Optional[int] = 0.5,
    figsize: Optional[tuple] = (1,10),
    cmap: Optional[str] = 'coolwarm',
    cbar: Optional[bool] = True
):

    if type not in ["observations", "signatures"]:
        raise ValueError('The "type" variable should be one of ["observations", "signatures"].')
    
    type_str = 'vision_obs_df_scores' if type == "observations" else 'vision_signature_scores'

    masked_data = adata.uns[type_str][['c_prime']].where((adata.uns[type_str][['fdr']] < 0.05).values)
    masked_data = masked_data.sort_values('c_prime', ascending=False)
    masked_data.columns = ['Consistency']

    plt.figure(figsize=figsize)
    sns.heatmap(masked_data, annot=masked_data, cmap=cmap, fmt='.2f', cbar=cbar, center=center)
    plt.show()

    return


def plot_vision_de_results(
    adata,
    type: Optional[Union[Literal["observations"], Literal["signatures"]]] = None,
    var: str = None,
    center: Optional[int] = 0.5,
    figsize: Optional[tuple] = (3,10),
    cmap: Optional[str] = 'coolwarm',
    cbar: Optional[bool] = True
):

    if var is None:
        raise ValueError('The "var" variable should be a categorical variable to plot.')
    
    if type not in ["observations", "signatures"]:
        raise ValueError('The "type" variable should be one of ["observations", "signatures"].')
    
    type_score_str = f'one_vs_all_obs_cols_{var}_scores' if type == "observations" else f'one_vs_all_signatures_{var}_scores'
    type_pval_str = f'one_vs_all_obs_cols_{var}_pvals' if type == "observations" else f'one_vs_all_signatures_{var}_padj'

    mask = adata.uns[type_pval_str] < 0.05

    plt.figure(figsize=figsize)
    sns.heatmap(adata.uns[type_score_str], mask=~mask, cmap=cmap, annot=mask.applymap(lambda x: '*' if x else ''), fmt='', cbar=cbar, center=center)
    plt.show()

    return


def plot_sig_mod_correlation(
    adata,
    x_rotation: Optional[int] = 0,
    y_rotation: Optional[int] = 0,
    use_FDR: Optional[bool] = True,
    subset_signatures: Optional[list] = None,
    subset_modules: Optional[list] = None,
    cmap: Optional[str] = 'RdBu_r',
):
    
    coef = adata.uns['sig_mod_correlation_coefs'] if 'sig_mod_correlation_coefs' in adata.uns.keys() else None
    if use_FDR:
        padj = adata.uns['sig_mod_correlation_FDR'] if 'sig_mod_correlation_FDR' in adata.uns.keys() else None
    else:
        padj = adata.uns['sig_mod_correlation_pvals'] if 'sig_mod_correlation_pvals' in adata.uns.keys() else None

    if coef is None or padj is None:
        raise ValueError('Run the "harreman.hs.integrate_vision_hotspot_results" function before plotting the results.')
    
    coef = coef.loc[subset_signatures] if subset_signatures is not None else coef
    padj = padj.loc[subset_signatures] if subset_signatures is not None else padj
    coef = coef[subset_modules] if subset_modules is not None else coef
    padj = padj[subset_modules] if subset_modules is not None else padj
    
    coef = coef[padj < 0.05].dropna(how='all').copy()
    padj = padj[padj < 0.05].dropna(how='all').copy()

    coef.replace(np.nan, 0, inplace=True)
    padj.replace(np.nan, 1, inplace=True)
    
    cmap = mpl.colormaps.get_cmap(cmap)
    cmap.set_bad("gray")

    g = sns.clustermap(coef, cmap=cmap, xticklabels=True, yticklabels=True, mask=padj > 0.05, center=0)

    fig = plt.gcf()

    for tick in g.ax_heatmap.get_xticklabels():
        tick.set_rotation(x_rotation)
    for tick in g.ax_heatmap.get_yticklabels():
        tick.set_rotation(y_rotation)

    padj = padj[g.data2d.columns]

    for i, ix in enumerate(g.dendrogram_row.reordered_ind):
        for j in range(len(coef.columns)):
                text = g.ax_heatmap.text(
                    j + 0.5,
                    i + 0.5,
                    "***" if padj.iloc[ix, j] < 0.0005 else "**"
                    if padj.iloc[ix, j] < 0.005 else "*" if padj.iloc[ix, j] < 0.05 else '',
                    ha="center",
                    va="center",
                    color="black",
                )
    
    # Find the colorbar 'child' and modify
    min_delta = 1e99
    min_aa = None
    for aa in fig.get_children():
        try:
            bbox = aa.get_position()
            delta = (0-bbox.xmin)**2 + (1-bbox.ymax)**2
            if delta < min_delta:
                delta = min_delta
                min_aa = aa
        except AttributeError:
            pass

    label = 'Spearman R' if adata.uns['cor_method'] == 'spearman' else 'Pearson R'
    min_aa.set_ylabel(label)
    min_aa.yaxis.set_label_position("left")
    
    plt.show()


def plot_sig_mod_enrichment(
    adata,
    x_rotation: Optional[int] = 0,
    y_rotation: Optional[int] = 0,
    use_FDR: Optional[bool] = True,
    subset_signatures: Optional[list] = None,
    subset_modules: Optional[list] = None,
    cmap: Optional[str] = 'RdBu_r',
):
    
    coef = adata.uns['sig_mod_enrichment_stats'] if 'sig_mod_enrichment_stats' in adata.uns.keys() else None
    if use_FDR:
        padj = adata.uns['sig_mod_enrichment_FDR'] if 'sig_mod_enrichment_FDR' in adata.uns.keys() else None
    else:
        padj = adata.uns['sig_mod_enrichment_pvals'] if 'sig_mod_enrichment_pvals' in adata.uns.keys() else None

    if coef is None or padj is None:
        raise ValueError('Run the "harreman.hs.integrate_vision_hotspot_results" function before plotting the results.')
    
    coef = coef.loc[subset_signatures] if subset_signatures is not None else coef
    padj = padj.loc[subset_signatures] if subset_signatures is not None else padj
    coef = coef[subset_modules] if subset_modules is not None else coef
    padj = padj[subset_modules] if subset_modules is not None else padj
    
    coef = coef[padj < 0.05].T.dropna(how='all').copy()
    padj = padj[padj < 0.05].T.dropna(how='all').copy()

    coef.replace(np.nan, 0, inplace=True)
    padj.replace(np.nan, 1, inplace=True)
    
    cmap = mpl.colormaps.get_cmap(cmap)
    cmap.set_bad("gray")

    g = sns.clustermap(coef, cmap=cmap, xticklabels=True, yticklabels=True, mask=padj > 0.05, center=0)

    for tick in g.ax_heatmap.get_xticklabels():
        tick.set_rotation(x_rotation)
    for tick in g.ax_heatmap.get_yticklabels():
        tick.set_rotation(y_rotation)

    padj = padj[g.data2d.columns]

    for i, ix in enumerate(g.dendrogram_row.reordered_ind):
        for j in range(len(coef.columns)):
                text = g.ax_heatmap.text(
                    j + 0.5,
                    i + 0.5,
                    "***" if padj.iloc[ix, j] < 0.0005 else "**"
                    if padj.iloc[ix, j] < 0.005 else "*" if padj.iloc[ix, j] < 0.05 else '',
                    ha="center",
                    va="center",
                    color="black",
                )


def plot_interaction_module_correlation(
    adata,
    x_rotation: Optional[int] = 0,
    y_rotation: Optional[int] = 0,
    use_FDR: Optional[bool] = True,
    subset_interactions: Optional[list] = None,
    subset_modules: Optional[list] = None,
    cmap: Optional[str] = 'RdBu_r',
    figsize: Optional[tuple] = (10,10),
    threshold: Optional[float] = None,
):
    
    coef = adata.uns['interaction_module_correlation_coefs'].T if 'interaction_module_correlation_coefs' in adata.uns.keys() else None
    if use_FDR:
        padj = adata.uns['interaction_module_correlation_FDR'].T if 'interaction_module_correlation_FDR' in adata.uns.keys() else None
    else:
        padj = adata.uns['interaction_module_correlation_pvals'].T if 'interaction_module_correlation_pvals' in adata.uns.keys() else None

    if coef is None or padj is None:
        raise ValueError('Run the "harreman.tl.compute_interaction_module_correlation" function before plotting the results.')
    
    coef = coef.loc[subset_interactions] if subset_interactions is not None else coef
    padj = padj.loc[subset_interactions] if subset_interactions is not None else padj
    coef = coef[subset_modules] if subset_modules is not None else coef
    padj = padj[subset_modules] if subset_modules is not None else padj
    
    coef = coef[padj < 0.05].dropna(how='all').copy()
    padj = padj[padj < 0.05].dropna(how='all').copy()

    coef.replace(np.nan, 0, inplace=True)
    padj.replace(np.nan, 1, inplace=True)
    
    if threshold:
        padj = padj[(coef > threshold).any(axis=1)]
        coef = coef[(coef > threshold).any(axis=1)]
    
    cmap = mpl.colormaps.get_cmap(cmap)
    cmap.set_bad("gray")

    g = sns.clustermap(coef, cmap=cmap, xticklabels=True, yticklabels=True, mask=padj > 0.05, center=0, figsize=figsize)

    fig = plt.gcf()

    for tick in g.ax_heatmap.get_xticklabels():
        tick.set_rotation(x_rotation)
    for tick in g.ax_heatmap.get_yticklabels():
        tick.set_rotation(y_rotation)

    padj = padj[g.data2d.columns]

    for i, ix in enumerate(g.dendrogram_row.reordered_ind):
        for j in range(len(coef.columns)):
            text = g.ax_heatmap.text(
                j + 0.5,
                i + 0.5,
                "***" if padj.iloc[ix, j] < 0.0005 else "**"
                if padj.iloc[ix, j] < 0.005 else "*" if padj.iloc[ix, j] < 0.05 else '',
                ha="center",
                va="center",
                color="black",
            )
    
    # Find the colorbar 'child' and modify
    min_delta = 1e99
    min_aa = None
    for aa in fig.get_children():
        try:
            bbox = aa.get_position()
            delta = (0-bbox.xmin)**2 + (1-bbox.ymax)**2
            if delta < min_delta:
                delta = min_delta
                min_aa = aa
        except AttributeError:
            pass

    label = 'Spearman R' if adata.uns['cor_method'] == 'spearman' else 'Pearson R'
    min_aa.set_ylabel(label)
    min_aa.yaxis.set_label_position("left")
    
    plt.show()


# The code below has been replicated from cell2location (https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html)

def get_rgb_function(cmap, min_value, max_value):
    r"""Generate a function to map continous values to RGB values using colormap between min_value & max_value."""

    if min_value > max_value:
        raise ValueError("Max_value should be greater or than min_value.")

    if min_value == max_value:
        warnings.warn(
            "Max_color is equal to min_color. It might be because of the data or bad parameter choice. "
            "If you are using plot_contours function try increasing max_color_quantile parameter and"
            "removing cell types with all zero values."
        )

        def func_equal(x):
            factor = 0 if max_value == 0 else 0.5
            return cmap(np.ones_like(x) * factor)

        return func_equal

    def func(x):
        return cmap((np.clip(x, min_value, max_value) - min_value) / (max_value - min_value))

    return func


def rgb_to_ryb(rgb):
    """
    Converts colours from RGB colorspace to RYB

    Parameters
    ----------

    rgb
        numpy array Nx3

    Returns
    -------
    Numpy array Nx3
    """
    rgb = np.array(rgb)
    if len(rgb.shape) == 1:
        rgb = rgb[np.newaxis, :]

    white = rgb.min(axis=1)
    black = (1 - rgb).min(axis=1)
    rgb = rgb - white[:, np.newaxis]

    yellow = rgb[:, :2].min(axis=1)
    ryb = np.zeros_like(rgb)
    ryb[:, 0] = rgb[:, 0] - yellow
    ryb[:, 1] = (yellow + rgb[:, 1]) / 2
    ryb[:, 2] = (rgb[:, 2] + rgb[:, 1] - yellow) / 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = ryb[mask].max(axis=1) / rgb[mask].max(axis=1)
        ryb[mask] = ryb[mask] / norm[:, np.newaxis]

    return ryb + black[:, np.newaxis]


def ryb_to_rgb(ryb):
    """
    Converts colours from RYB colorspace to RGB

    Parameters
    ----------

    ryb
        numpy array Nx3

    Returns
    -------
    Numpy array Nx3
    """
    ryb = np.array(ryb)
    if len(ryb.shape) == 1:
        ryb = ryb[np.newaxis, :]

    black = ryb.min(axis=1)
    white = (1 - ryb).min(axis=1)
    ryb = ryb - black[:, np.newaxis]

    green = ryb[:, 1:].min(axis=1)
    rgb = np.zeros_like(ryb)
    rgb[:, 0] = ryb[:, 0] + ryb[:, 1] - green
    rgb[:, 1] = green + ryb[:, 1]
    rgb[:, 2] = (ryb[:, 2] - green) * 2

    mask = ~(ryb == 0).all(axis=1)
    if mask.any():
        norm = rgb[mask].max(axis=1) / ryb[mask].max(axis=1)
        rgb[mask] = rgb[mask] / norm[:, np.newaxis]

    return rgb + white[:, np.newaxis]


def plot_spatial_general(
    value_df,
    coords,
    labels,
    text=None,
    circle_diameter=4.0,
    alpha_scaling=1.0,
    max_col=(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
    max_color_quantile=0.98,
    show_img=True,
    img=None,
    img_alpha=1.0,
    adjust_text=False,
    plt_axis="off",
    axis_y_flipped=True,
    x_y_labels=("", ""),
    crop_x=None,
    crop_y=None,
    text_box_alpha=0.9,
    reorder_cmap=range(7),
    style="fast",
    colorbar_position="bottom",
    colorbar_label_kw={},
    colorbar_shape={},
    colorbar_tick_size=12,
    colorbar_grid=None,
    image_cmap="Greys_r",
    white_spacing=20,
):
    r"""Plot spatial abundance of cell types (regulatory programmes) with colour gradient and interpolation.

      This method supports only 7 cell types with these colours (in order, which can be changed using reorder_cmap).
      'yellow' 'orange' 'blue' 'green' 'purple' 'grey' 'white'

    :param value_df: pd.DataFrame - with cell abundance or other features (only 7 allowed, columns) across locations (rows)
    :param coords: np.ndarray - x and y coordinates (in columns) to be used for ploting spots
    :param text: pd.DataFrame - with x, y coordinates, text to be printed
    :param circle_diameter: diameter of circles
    :param labels: list of strings, labels of cell types
    :param alpha_scaling: adjust color alpha
    :param max_col: crops the colorscale maximum value for each column in value_df.
    :param max_color_quantile: crops the colorscale at x quantile of the data.
    :param show_img: show image?
    :param img: numpy array representing a tissue image.
        If not provided a black background image is used.
    :param img_alpha: transparency of the image
    :param lim: x and y max limits on the plot. Minimum is always set to 0, if `lim` is None maximum
        is set to image height and width. If 'no_limit' then no limit is set.
    :param adjust_text: move text label to prevent overlap
    :param plt_axis: show axes?
    :param axis_y_flipped: flip y axis to match coordinates of the plotted image
    :param reorder_cmap: reorder colors to make sure you get the right color for each category

    :param style: plot style (matplolib.style.context):
        'fast' - white background & dark text;
        'dark_background' - black background & white text;

    :param colorbar_position: 'bottom', 'right' or None
    :param colorbar_label_kw: dict that will be forwarded to ax.set_label()
    :param colorbar_shape: dict {'vertical_gaps': 1.5, 'horizontal_gaps': 1.5,
                                    'width': 0.2, 'height': 0.2}, not obligatory to contain all params
    :param colorbar_tick_size: colorbar ticks label size
    :param colorbar_grid: tuple of colorbar grid (rows, columns)
    :param image_cmap: matplotlib colormap for grayscale image
    :param white_spacing: percent of colorbars to be hidden

    """

    # if value_df.shape[1] > 7:
    #     raise ValueError("Maximum of 7 cell types / factors can be plotted at the moment")

    def create_colormap(R, G, B):
        spacing = int(white_spacing * 2.55)

        N = 255
        M = 3

        alphas = np.concatenate([[0] * spacing * M, np.linspace(0, 1.0, (N - spacing) * M)])

        vals = np.ones((N * M, 4))
        #         vals[:, 0] = np.linspace(1, R / 255, N * M)
        #         vals[:, 1] = np.linspace(1, G / 255, N * M)
        #         vals[:, 2] = np.linspace(1, B / 255, N * M)
        for i, color in enumerate([R, G, B]):
            vals[:, i] = color / 255
        vals[:, 3] = alphas

        return ListedColormap(vals)

    # # Create linearly scaled colormaps
    # YellowCM = create_colormap(240, 228, 66)  # #F0E442 
    # RedCM = create_colormap(213, 94, 0)  # #D55E00
    # BlueCM = create_colormap(86, 180, 233)  # #56B4E9
    # GreenCM = create_colormap(0, 158, 115)  # #009E73
    # GreyCM = create_colormap(200, 200, 200)  # #C8C8C8
    # WhiteCM = create_colormap(50, 50, 50)  # #323232
    # PurpleCM = create_colormap(90, 20, 165)  # #5A14A5

    # Create linearly scaled colormaps
    BlueCM = create_colormap(31, 119, 180)  #1f77b4 
    OrangeCM = create_colormap(255, 127, 14)  #ff7f0e
    GreenCM = create_colormap(44, 160, 44)  #2ca02c
    RedCM = create_colormap(214, 39, 40)  #d62728
    PurpleCM = create_colormap(148, 103, 189)  #9467bd
    BrownCM = create_colormap(140, 86, 75)  #8c564b
    PinkCM = create_colormap(227, 119, 194)  #e377c2
    GreyCM = create_colormap(127, 127, 127)  #7f7f7f
    YellowCM = create_colormap(188, 189, 34)  #bcbd22
    LightBlueCM = create_colormap(23, 190, 207)  #17becf
    BlackCM = create_colormap(50, 50, 50)  #323232

    # cmaps = [RedCM, BlueCM, YellowCM, GreenCM, PurpleCM, GreyCM, WhiteCM]
    cmaps = [BlueCM, OrangeCM, GreenCM, RedCM, PurpleCM, BrownCM, PinkCM, GreyCM, YellowCM, LightBlueCM, BlackCM]

    reorder_cmap = range(len(cmaps))
    cmaps = [cmaps[i] for i in reorder_cmap]

    with mpl.style.context(style):
        fig = plt.figure()

        if colorbar_position == "right":
            if colorbar_grid is None:
                colorbar_grid = (len(labels), 1)

            shape = {"vertical_gaps": 1.5, "horizontal_gaps": 0, "width": 0.15, "height": 0.2}
            shape = {**shape, **colorbar_shape}

            gs = GridSpec(
                nrows=colorbar_grid[0] + 2,
                ncols=colorbar_grid[1] + 1,
                width_ratios=[1, *[shape["width"]] * colorbar_grid[1]],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0], 1],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )
            ax = fig.add_subplot(gs[:, 0], aspect="equal", rasterized=True)

        if colorbar_position == "bottom":
            if colorbar_grid is None:
                if len(labels) <= 3:
                    colorbar_grid = (1, len(labels))
                else:
                    n_rows = round(len(labels) / 3 + 0.5 - 1e-9)
                    colorbar_grid = (n_rows, 3)

            shape = {"vertical_gaps": 0.3, "horizontal_gaps": 0.6, "width": 0.2, "height": 0.035}
            shape = {**shape, **colorbar_shape}

            gs = GridSpec(
                nrows=colorbar_grid[0] + 1,
                ncols=colorbar_grid[1] + 2,
                width_ratios=[0.3, *[shape["width"]] * colorbar_grid[1], 0.3],
                height_ratios=[1, *[shape["height"]] * colorbar_grid[0]],
                hspace=shape["vertical_gaps"],
                wspace=shape["horizontal_gaps"],
            )

            ax = fig.add_subplot(gs[0, :], aspect="equal", rasterized=True)

        if colorbar_position is None:
            ax = fig.add_subplot(aspect="equal", rasterized=True)

        if colorbar_position is not None:
            cbar_axes = []
            for row in range(1, colorbar_grid[0] + 1):
                for column in range(1, colorbar_grid[1] + 1):
                    cbar_axes.append(fig.add_subplot(gs[row, column]))

            n_excess = colorbar_grid[0] * colorbar_grid[1] - len(labels)
            if n_excess > 0:
                for i in range(1, n_excess + 1):
                    cbar_axes[-i].set_visible(False)

        ax.set_xlabel(x_y_labels[0])
        ax.set_ylabel(x_y_labels[1])

        if img is not None and show_img:
            ax.imshow(img, aspect="equal", alpha=img_alpha, origin="lower", cmap=image_cmap)

        # crop images in needed
        if crop_x is not None:
            ax.set_xlim(crop_x[0], crop_x[1])
        if crop_y is not None:
            ax.set_ylim(crop_y[0], crop_y[1])

        if axis_y_flipped:
            ax.invert_yaxis()

        if plt_axis == "off":
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        counts = value_df.values.copy()

        # plot spots as circles
        c_ord = list(np.arange(0, counts.shape[1]))

        colors = np.zeros((*counts.shape, 4))
        weights = np.zeros(counts.shape)
        
        max_col = tuple([np.inf for i in range(len(cmaps))])

        for c in c_ord:
            min_color_intensity = counts[:, c].min()
            max_color_intensity = np.min([np.quantile(counts[:, c], max_color_quantile), max_col[c]])

            rgb_function = get_rgb_function(cmap=cmaps[c], min_value=min_color_intensity, max_value=max_color_intensity)

            color = rgb_function(counts[:, c])
            color[:, 3] = color[:, 3] * alpha_scaling

            norm = mpl.colors.Normalize(vmin=min_color_intensity, vmax=max_color_intensity)

            if colorbar_position is not None:
                cbar_ticks = [
                    min_color_intensity,
                    np.mean([min_color_intensity, max_color_intensity]),
                    max_color_intensity,
                ]
                cbar_ticks = np.array(cbar_ticks)

                if max_color_intensity > 13:
                    cbar_ticks = cbar_ticks.astype(np.int32)
                else:
                    cbar_ticks = cbar_ticks.round(2)

                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmaps[c]),
                    cax=cbar_axes[c],
                    orientation="horizontal",
                    extend="both",
                    ticks=cbar_ticks,
                )

                cbar.ax.tick_params(labelsize=colorbar_tick_size)
                max_color = rgb_function(max_color_intensity / 1.5)
                cbar.ax.set_title(labels[c], **{**{"size": 20, "color": max_color, "alpha": 1}, **colorbar_label_kw})

            colors[:, c] = color
            weights[:, c] = np.clip(counts[:, c] / (max_color_intensity + 1e-10), 0, 1)
            weights[:, c][counts[:, c] < min_color_intensity] = 0

        colors_ryb = np.zeros((*weights.shape, 3))

        for i in range(colors.shape[0]):
            colors_ryb[i] = rgb_to_ryb(colors[i, :, :3])

        def kernel(w):
            return w**2

        kernel_weights = kernel(weights[:, :, np.newaxis])
        weighted_colors_ryb = (colors_ryb * kernel_weights).sum(axis=1) / kernel_weights.sum(axis=1)

        weighted_colors = np.zeros((weights.shape[0], 4))

        weighted_colors[:, :3] = ryb_to_rgb(weighted_colors_ryb)

        weighted_colors[:, 3] = colors[:, :, 3].max(axis=1)

        ax.scatter(x=coords[:, 0], y=coords[:, 1], c=weighted_colors, s=circle_diameter**2)

        # add text
        if text is not None:
            bbox_props = dict(boxstyle="round", ec="0.5", alpha=text_box_alpha, fc="w")
            texts = []
            for x, y, s in zip(
                np.array(text.iloc[:, 0].values).flatten(),
                np.array(text.iloc[:, 1].values).flatten(),
                text.iloc[:, 2].tolist(),
            ):
                texts.append(ax.text(x, y, s, ha="center", va="bottom", bbox=bbox_props))

            if adjust_text:
                from adjustText import adjust_text

                adjust_text(texts, arrowprops=dict(arrowstyle="->", color="w", lw=0.5))

    return fig


def plot_spatial(adata, color, img_key="hires", show_img=True, **kwargs):
    """Plot spatial abundance of cell types (regulatory programmes) with colour gradient
    and interpolation (from Visium anndata).

    This method supports only 7 cell types with these colours (in order, which can be changed using reorder_cmap).
    'yellow' 'orange' 'blue' 'green' 'purple' 'grey' 'white'

    :param adata: adata object with spatial coordinates in adata.obsm['spatial']
    :param color: list of adata.obs column names to be plotted
    :param kwargs: arguments to plot_spatial_general
    :return: matplotlib figure
    """

    if show_img is True:
        kwargs["show_img"] = True
        kwargs["img"] = list(adata.uns["spatial"].values())[0]["images"][img_key]

    # location coordinates
    if "spatial" in adata.uns.keys():
        kwargs["coords"] = (
            adata.obsm["spatial"] * list(adata.uns["spatial"].values())[0]["scalefactors"][f"tissue_{img_key}_scalef"]
        )
    else:
        kwargs["coords"] = adata.obsm["spatial"]

    fig = plot_spatial_general(value_df=adata.obs[color], **kwargs)  # cell abundance values

    return fig
