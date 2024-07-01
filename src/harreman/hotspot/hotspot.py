import math
from typing import Literal, Optional, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.sparse import issparse
from scipy.stats import chisquare, pearsonr

from .diffexp import rank_genes_groups
from .modules import compute_sig_mod_enrichment
from .signature import compute_obs_df_scores, compute_signature_scores
from .utils import filter_genes


class Hotspot:
    def __init__(
        self,
        adata: Optional[anndata.AnnData] = None,
        norm_data_key: Optional[Union[Literal["use_raw"], str]] = None,
        compute_neighbors_on_key: Optional[str] = None,
        distances_obsp_key: Optional[str] = None,
        signature_varm_key: Optional[str] = None,
        signature_names_uns_key: Optional[str] = None,
        weighted_graph: Optional[bool] = True,
        n_neighbors: Optional[int] = None,
        neighborhood_factor: Optional[int] = 3,
        sample_key: Optional[str] = None,
        layer_key: Optional[Union[Literal["use_raw"], str]] = None,
        model: Optional[str] = None,
        neighborhood_radius: Optional[int] = 100,
        jobs: Optional[int] = None,
    ) -> None:
        """Accessor to AnnData object.

        Parameters
        ----------
        adata
            AnnData object
        norm_data_key
            Key for layer with log library size normalized data. If
            `None` (default), uses `adata.X`
        protein_obsm_key
            Location for protein data
        signature_varm_key
            Location for genes by signature matrix

        """
        self._adata = adata
        self._norm_data_key = norm_data_key
        self._compute_neighbors_on_key = compute_neighbors_on_key
        self._distances_obsp_key = distances_obsp_key
        self._signature_varm_key = signature_varm_key
        self._signature_names_uns_key = signature_names_uns_key
        self._weighted_graph = weighted_graph
        self._n_neighbors = n_neighbors
        self._neighborhood_factor = neighborhood_factor
        self._sample_key = sample_key
        self._layer_key = layer_key
        self._model = model
        self._neighborhood_radius = neighborhood_radius
        self._jobs = jobs

    @property
    def adata(self):
        return self._adata

    @adata.setter
    def adata(self, key: str):
        self._adata = key

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

    @property
    def min_gene_threshold(self):
        return self._min_gene_threshold

    @min_gene_threshold.setter
    def min_gene_threshold(self, key: int):
        self._min_gene_threshold = key

    @property
    def core_only(self):
        return self._core_only

    @core_only.setter
    def core_only(self, key: int):
        self._core_only = key

    @property
    def fdr_threshold(self):
        return self._fdr_threshold

    @fdr_threshold.setter
    def fdr_threshold(self, key: int):
        self._fdr_threshold = key

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

    def filter_genes(self):
        self.adata.uns['filtered_genes'], self.adata.uns['filtered_genes_ct'] = filter_genes(self.adata, self.layer_key, self.database_varm_key, self.cell_type_key, self.expression_filt, self.de_filt, self.autocorrelation_filt)

    def compute_one_vs_all_signatures(self):
        sig_adata = anndata.AnnData(self.adata.obsm["vision_signatures"])
        sig_adata.obs = self.adata.obs.loc[:, self.cat_obs_cols].copy()
        for c in self.cat_obs_cols:
            rank_genes_groups(
                sig_adata,
                groupby=c,
                key_added=f"rank_genes_groups_{c}",
                method="wilcoxon",
            )
        self.sig_adata = sig_adata

    def compute_one_vs_all_obs_cols(self):
        # log for scanpy de
        obs_adata = anndata.AnnData(np.log1p(self.adata.obs._get_numeric_data().copy()))
        obs_adata.obs = self.adata.obs.loc[:, self.cat_obs_cols].copy()
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

        self.obs_adata = obs_adata

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

    def compute_sig_mod_enrichment(self):
        pvals_df, stats_df, FDR_df = compute_sig_mod_enrichment(self.adata, self.norm_data_key, self.signature_varm_key)
        self.adata.uns["sig_mod_enrichment_stats"] = stats_df
        self.adata.uns["sig_mod_enrichment_pvals"] = pvals_df
        self.adata.uns["sig_mod_enrichment_FDR"] = FDR_df


def categories(col):
    return col.astype("category").cat.categories
