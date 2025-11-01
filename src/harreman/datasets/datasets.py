
import os
import tempfile
from pathlib import Path
import scanpy as sc

HERE = Path(__file__).resolve()
BASE = HERE.parents[3] / "data" / "h5ads"


def load_visium_mouse_colon_dataset() -> "sc.AnnData":
    """
    Load the mouse colon Visium dataset.

    Returns
    -------
    adata : AnnData
        The loaded Visium dataset.
    """

    adata_path = f"{BASE}/Visium_colon_unrolled_adata.h5ad"
    adata = sc.read_h5ad(adata_path)

    return adata


def load_slide_seq_human_lung_dataset() -> "sc.AnnData":
    """
    Load the human lung Slide-seq dataset.

    Returns
    -------
    adata : AnnData
        The loaded Slide-seq dataset.
    """

    adata_path = f"{BASE}/Slide_seq_lung_adata.h5ad"
    adata = sc.read_h5ad(adata_path)

    return adata
