import math
from typing import Literal, Optional, Union
import time
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.sparse import issparse
from scipy.stats import chisquare, pearsonr

from .diffexp import rank_genes_groups
from .signature import compute_obs_df_scores, compute_signature_scores


class VISION:
    def __init__(
        self,
        adata: Optional[anndata.AnnData] = None,
        norm_data_key: Optional[Union[Literal["use_raw"], str]] = None,
        compute_neighbors_on_key: Optional[str] = None,
        distances_obsp_key: Optional[str] = None,
        signature_varm_key: Optional[str] = None,
        signature_names_uns_key: Optional[str] = None,
        weighted_graph: Optional[bool] = True,
        neighborhood_radius: Optional[int] = 100,
        n_neighbors: Optional[int] = 30,
        neighborhood_factor: Optional[int] = 3,
        sample_key: Optional[str] = None,
    ) -> None:
        """VISION object.

        Parameters
        ----------
        adata
            AnnData object
        norm_data_key
            Key for layer with log library size normalized data. If
            `None` (default), uses `adata.X`
        compute_neighbors_on_key
            Key in `adata.obsm` to use for computing neighbors. If `None`, use neighbors stored in `adata`. If no neighbors have been previously computed an error will be raised.
        distances_obsp_key
            Distances encoding cell-cell similarities directly. Shape is (cells x cells). Input is key in `adata.obsp`.
        signature_varm_key
            Location for genes by signature matrix
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

        """
        self._adata = adata
        self._norm_data_key = norm_data_key
        self._compute_neighbors_on_key = compute_neighbors_on_key
        self._distances_obsp_key = distances_obsp_key
        self._signature_varm_key = signature_varm_key
        self._signature_names_uns_key = signature_names_uns_key
        self._weighted_graph = weighted_graph
        self._neighborhood_radius = neighborhood_radius
        self._n_neighbors = n_neighbors
        self._neighborhood_factor = neighborhood_factor
        self._sample_key = sample_key
        self._cells_selections = {}

    @property
    def adata(self):
        return self._adata

    @property
    def var_names(self):
        if self._norm_data_key == "use_raw":
            return self.adata.raw.var_names
        else:
            return self.adata.var_names

    @property
    def cells_selections(self):
        return self._cells_selections.keys()

    def add_cells_selection(self, key, val):
        self._cells_selections[key] = val

    def get_cells_selection(self, key):
        return self._cells_selections[key]

    @adata.setter
    def adata(self, adata: anndata.AnnData):
        self._adata = adata
        num_cols = adata.obs._get_numeric_data().columns.tolist()
        cols = adata.obs.columns.tolist()
        cat_vars = list(set(cols) - set(num_cols))
        self.cat_obs_cols = cat_vars
        self.numeric_obs_cols = num_cols

    @property
    def norm_data_key(self):
        return self._norm_data_key

    @norm_data_key.setter
    def norm_data_key(self, key: str):
        self._norm_data_key = key

    @property
    def compute_neighbors_on_key(self):
        return self._compute_neighbors_on_key

    @compute_neighbors_on_key.setter
    def compute_neighbors_on_key(self, key: str):
        self._compute_neighbors_on_key = key

    @property
    def distances_obsp_key(self):
        return self._distances_obsp_key

    @distances_obsp_key.setter
    def distances_obsp_key(self, key: str):
        self._distances_obsp_key = key

    @property
    def signature_varm_key(self):
        return self._signature_varm_key

    @signature_varm_key.setter
    def signature_varm_key(self, key: str):
        self._signature_varm_key = key

    @property
    def signature_names_uns_key(self):
        return self._signature_names_uns_key

    @signature_names_uns_key.setter
    def signature_names_uns_key(self, key: str):
        self._signature_names_uns_key = key

    @property
    def weighted_graph(self):
        return self._weighted_graph

    @weighted_graph.setter
    def weighted_graph(self, key: list):
        self._weighted_graph = key

    @property
    def n_neighbors(self):
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, key: list):
        self._n_neighbors = key

    @property
    def neighborhood_factor(self):
        return self._neighborhood_factor

    @neighborhood_factor.setter
    def neighborhood_factor(self, key: list):
        self._neighborhood_factor = key

    @property
    def sample_key(self):
        return self._sample_key

    @sample_key.setter
    def sample_key(self, key: str):
        self._sample_key = key

    @property
    def obs_df_scores(self):
        return self._obs_df_scores

    @obs_df_scores.setter
    def obs_df_scores(self, key: str):
        self._obs_df_scores = key

    @property
    def one_vs_all_obs_cols(self):
        return self._one_vs_all_obs_cols

    @one_vs_all_obs_cols.setter
    def one_vs_all_obs_cols(self, key: str):
        self._one_vs_all_obs_cols = key

    @property
    def one_vs_all_signatures(self):
        return self._one_vs_all_signatures

    @one_vs_all_signatures.setter
    def one_vs_all_signatures(self, key: str):
        self._one_vs_all_signatures = key

    @property
    def gene_score_per_signature(self):
        return self._gene_score_per_signature

    @gene_score_per_signature.setter
    def gene_score_per_signature(self, key: str):
        self._gene_score_per_signature = key

    @property
    def layer_key(self):
        return self._layer_key

    @layer_key.setter
    def layer_key(self, key: str):
        self._layer_key = key

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, key: str):
        self._model = key

    @property
    def neighborhood_radius(self):
        return self._neighborhood_radius

    @neighborhood_radius.setter
    def neighborhood_radius(self, key: list):
        self._neighborhood_radius = key

    @property
    def jobs(self):
        return self._jobs

    @jobs.setter
    def jobs(self, key: int):
        self._jobs = key

    @property
    def deconv_data(self):
        return self._deconv_data

    @deconv_data.setter
    def deconv_data(self, key: bool):
        self._deconv_data = key

    @property
    def cell_type_list(self):
        return self._cell_type_list

    @cell_type_list.setter
    def cell_type_list(self, key: list):
        self._cell_type_list = key

    @property
    def cell_type_key(self):
        return self._cell_type_key

    @cell_type_key.setter
    def cell_type_key(self, key: str):
        self._cell_type_key = key

    @property
    def cell_type_pairs(self):
        return self._cell_type_pairs

    @cell_type_pairs.setter
    def cell_type_pairs(self, key: str):
        self._cell_type_pairs = key

    @property
    def database_varm_key(self):
        return self._database_varm_key

    @database_varm_key.setter
    def database_varm_key(self, key: str):
        self._database_varm_key = key

    @property
    def spot_diameter(self):
        return self._spot_diameter

    @spot_diameter.setter
    def spot_diameter(self, key: int):
        self._spot_diameter = key

    @property
    def autocorrelation_filt(self):
        return self._autocorrelation_filt

    @autocorrelation_filt.setter
    def autocorrelation_filt(self, key: int):
        self._autocorrelation_filt = key

    @property
    def expression_filt(self):
        return self._expression_filt

    @expression_filt.setter
    def expression_filt(self, key: int):
        self._expression_filt = key

    @property
    def de_filt(self):
        return self._de_filt

    @de_filt.setter
    def de_filt(self, key: int):
        self._de_filt = key

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, key: int):
        self._test = key

    def get_gene_expression(self, gene: str, return_list=True) -> list:
        if self.adata is None:
            raise ValueError("Accessor not populated with anndata.")
        if self.norm_data_key == "use_raw":
            data = self.adata.raw[:, gene].X
        elif self.norm_data_key is None:
            data = self.adata[:, gene].X
        else:
            data = self.adata[:, gene].layers[self.norm_data_key]

        if scipy.sparse.issparse(data):
            data = data.toarray()

        if return_list:
            return data.ravel().tolist()
        else:
            return data

    def get_genes_by_signature(self, sig_name: str) -> pd.DataFrame:
        """Df of genes in index, sign as values."""

        if self.signature_names_uns_key is not None:
            index = np.where(np.asarray(self.adata.uns[self.signature_names_uns_key]) == sig_name)[0][0]
        else:
            index = np.where(np.asarray(self.adata.obsm["vision_signatures"].columns) == sig_name)[0][0]

        if self._norm_data_key == "use_raw":
            matrix = self.adata.raw.varm[self.signature_varm_key]
        else:
            matrix = self.adata.varm[self.signature_varm_key]

        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.to_numpy()

        matrix = matrix[:, index]
        if issparse(matrix):
            matrix = matrix.toarray().ravel()

        mask = matrix != 0
        sig_df = pd.DataFrame(index=self.var_names[mask], data=matrix[mask])

        return sig_df

    def compute_obs_df_scores(self):
        self.adata.uns["vision_obs_df_scores"] = compute_obs_df_scores(self.adata)

    def compute_signature_scores(self):
        self.adata.uns["vision_signature_scores"] = compute_signature_scores(self.adata, self.norm_data_key, self.signature_varm_key)

    def compute_one_vs_all_signatures(self):
        start = time.time()
        print("Computing one vs all DE analysis of the signature scores...")

        sig_adata = anndata.AnnData(self.adata.obsm["vision_signatures"])
        sig_adata.obs = self.adata.obs.loc[:, self.cat_obs_cols].copy()
        for c in self.cat_obs_cols:
            rank_genes_groups(
                sig_adata,
                groupby=c,
                key_added=f"rank_genes_groups_{c}",
                method="wilcoxon",
            )

            c_names = sig_adata.var_names.tolist()
            
            names_df = pd.DataFrame(sig_adata.uns[f'rank_genes_groups_{c}']['names'])
            scores_df = pd.DataFrame(sig_adata.uns[f'rank_genes_groups_{c}']['scores'])
            lfc_df = pd.DataFrame(sig_adata.uns[f'rank_genes_groups_{c}']['logfoldchanges'])
            padj_df = pd.DataFrame(sig_adata.uns[f'rank_genes_groups_{c}']['pvals_adj'])

            names_df, [scores_df, lfc_df, padj_df] = reorder_dataframes(names_df, [scores_df, lfc_df, padj_df], c_names)
            scores_df.index = c_names
            lfc_df.index = c_names
            padj_df.index = c_names

            self.adata.uns[f'one_vs_all_signatures_{c}_scores'] = scores_df
            self.adata.uns[f'one_vs_all_signatures_{c}_padj'] = padj_df

        self.sig_adata = sig_adata

        print("Finished computing one vs all DE analysis of the signature scores in %.3f seconds" %(time.time()-start))

        return

    def compute_one_vs_all_obs_cols(self):
        # log for scanpy de
        start = time.time()
        print("Computing one vs all DE analysis of the numerical data...")
        
        obs_adata = anndata.AnnData(np.log1p(self.adata.obs._get_numeric_data().copy()))
        obs_adata.obs = self.adata.obs.loc[:, self.cat_obs_cols].copy()
        colnames = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
        for c in self.cat_obs_cols:
            try:
                rank_genes_groups(
                    obs_adata,
                    groupby=c,
                    key_added=f"rank_genes_groups_{c}",
                    method="wilcoxon",
                )
            # one category only has one obs
            except ValueError:
                # TODO: Log it
                self.cat_obs_cols = [c_ for c_ in self.cat_obs_cols if c_ != c]
                continue

            for g in categories(obs_adata.obs[c]):
                mask = (obs_adata.obs[c] == g).to_numpy()
                obs_pos_masked = obs_adata.obs.iloc[mask]
                obs_neg_masked = obs_adata.obs.iloc[~mask]
                for j in obs_pos_masked.columns:
                    pos_freq = obs_pos_masked[j].value_counts(normalize=False)
                    neg_freq = obs_neg_masked[j].value_counts(normalize=False)
                    freqs = pd.concat([pos_freq, neg_freq], axis=1).fillna(0)
                    # TODO: cramer's v might be incorrect
                    grand_total = np.sum(freqs.to_numpy())
                    r = len(freqs) - 1
                    try:
                        stat, pval = chisquare(
                            freqs.iloc[:, 0].to_numpy().ravel(),
                            freqs.iloc[:, 1].to_numpy().ravel(),
                        )
                    except ValueError:
                        stat = grand_total * r  # so that v is 1
                        pval = 0
                    if math.isinf(pval) or math.isnan(pval):
                        pval = 1
                    if math.isinf(stat) or math.isnan(stat):
                        v = 1
                    else:
                        v = np.sqrt(stat / (grand_total * r))
                    obs_adata.uns[f"chi_sq_{j}_{g}"] = {
                        "stat": v,
                        "pval": pval,
                    }
            
            c_names = obs_adata.var_names.tolist()
            
            names_df = pd.DataFrame(obs_adata.uns[f'rank_genes_groups_{c}']['names'])
            scores_df = pd.DataFrame(obs_adata.uns[f'rank_genes_groups_{c}']['scores'])
            lfc_df = pd.DataFrame(obs_adata.uns[f'rank_genes_groups_{c}']['logfoldchanges'])
            padj_df = pd.DataFrame(obs_adata.uns[f'rank_genes_groups_{c}']['pvals_adj'])

            names_df, [scores_df, lfc_df, padj_df] = reorder_dataframes(names_df, [scores_df, lfc_df, padj_df], c_names)
            scores_df.index = c_names
            lfc_df.index = c_names
            padj_df.index = c_names

            cat_stat_df = pd.DataFrame(np.nan, index=self.cat_obs_cols, columns=categories(obs_adata.obs[c]).tolist())
            cat_pval_df = pd.DataFrame(np.nan, index=self.cat_obs_cols, columns=categories(obs_adata.obs[c]).tolist())

            for cat_col in self.cat_obs_cols:
                for cat in categories(obs_adata.obs[c]):
                    cat_stat_df.loc[cat_col, cat] = obs_adata.uns[f'chi_sq_{cat_col}_{cat}']['stat']
                    cat_pval_df.loc[cat_col, cat] = obs_adata.uns[f'chi_sq_{cat_col}_{cat}']['pval']
            
            scores_df_all = pd.concat([scores_df, cat_stat_df])
            pvals_df_all = pd.concat([padj_df, cat_pval_df])

            self.adata.uns[f'one_vs_all_obs_cols_{c}_scores'] = scores_df_all
            self.adata.uns[f'one_vs_all_obs_cols_{c}_pvals'] = pvals_df_all

        self.obs_adata = obs_adata

        print("Finished computing one vs all DE analysis of the numerical data in %.3f seconds" %(time.time()-start))

        return

    # TODO: refactor this function
    def compute_gene_score_per_signature(self):
        start = time.time()
        print("Computing gene score per signature...")

        gene_score_sig = {}

        if self.signature_names_uns_key is not None:
            sig_names = self.adata.uns[self.signature_names_uns_key]
        else:
            sig_names = self.adata.obsm["vision_signatures"].columns

        for s in sig_names:
            gene_score_sig[s] = {"genes": [], "values": []}
            df = self.get_genes_by_signature(s)
            gene_names = df.index
            # cells by genes
            expr = np.array(self.get_gene_expression(gene_names, return_list=False))
            # cells
            sign = df.to_numpy().ravel()
            gene_score_sig[s]["signs"] = sign.tolist()

            # TODO: Make faster
            for i, (g, sign_) in enumerate(zip(gene_names, sign)):
                gene_score_sig[s]["values"].append(
                    sign_ * pearsonr(expr[:, i], self.adata.obsm["vision_signatures"][s])[0]
                )
                gene_score_sig[s]["genes"].append(g)

        for s in sig_names:
            info = gene_score_sig[s]
            gene_score_sig[s]["geneImportance"] = {g: v for g, v in zip(info["genes"], info["values"])}
            gene_score_sig[s]["sigDict"] = {g: v for g, v in zip(info["genes"], info["signs"])}

        self.gene_score_sig = gene_score_sig

        print("Finished computing gene score per signature in %.3f seconds" %(time.time()-start))

        return

    def compute_one_vs_one_de(self, key: str, group1: str, group2: str):
        rank_genes_groups(
            self.adata,
            groupby=key,
            groups=[group1],
            reference=group2,
            key_added=f"rank_genes_groups_{key}",
            method="wilcoxon",
            use_raw=self.norm_data_key == "use_raw",
            layer=self.norm_data_key if self.norm_data_key != "use_raw" else None,
        )
        return sc.get.rank_genes_groups_df(self.adata, group1, key=f"rank_genes_groups_{key}")


def categories(col):
    return col.astype("category").cat.categories


def get_sort_index(column_values, order_list):
    return [order_list.index(x) for x in column_values]


def reorder_dataframes(main_df, dfs_to_reorder, order_list):
    reordered_main_df = main_df.copy()
    
    for col in main_df.columns:
        col_values = main_df[col].tolist()
        
        sort_index = [col_values.index(val) for val in order_list]
        
        reordered_main_df[col] = order_list
        
        for df in dfs_to_reorder:
            df[col] = [df[col][i] for i in sort_index]
    
    return reordered_main_df, dfs_to_reorder
