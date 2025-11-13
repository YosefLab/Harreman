# API

Import Harreman as:

```
import harreman
```

```{eval-rst}
.. currentmodule:: harreman

```

## Datasets

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   datasets.load_visium_mouse_colon_dataset
   datasets.load_slide_seq_human_lung_dataset
```

## Hotspot functions for metabolic zonation

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   hotspot.load_metabolic_genes
   hotspot.compute_local_autocorrelation
   hotspot.compute_local_correlation
   hotspot.create_modules
   hotspot.calculate_module_scores
   hotspot.compute_top_scoring_modules
   hotspot.calculate_super_module_scores
```

## Interaction database

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   database.extract_interaction_db
```

## Setting up deconvolution AnnData

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   anndata.setup_deconv_adata
```

## Metabolic crosstalk

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   tools.compute_knn_graph
   tools.apply_gene_filtering
   tools.compute_gene_pairs
   tools.compute_cell_communication
   tools.compute_ct_cell_communication
   tools.select_significant_interactions
   tools.compute_interacting_cell_scores
   tools.compute_ct_interacting_cell_scores
   tools.compute_interaction_module_correlation
```

## Visualization

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   plots.local_correlation_plot
   plots.average_local_correlation_plot
   plots.module_score_correlation_plot
   plots.plot_interacting_cell_scores
   plots.plot_ct_interacting_cell_scores
```

## Reading/writing functions

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   anndata.read_h5ad
   anndata.write_h5ad
```
