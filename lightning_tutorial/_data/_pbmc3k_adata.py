
__module_name__ = "_pbmc3k_adata.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# import packages: -------------------------------------------------------
import scanpy as sc
import pandas as pd


# primary function: ------------------------------------------------------
def _pbmc3k_adata(silent=False):

    """Use the scanpy API to load the pbmc3k dataset and format the louvain/celltype labels as integers."""

    adata = sc.datasets.pbmc3k_processed()

    # format louvain (cell type) labels as integers
    cell_type_df = pd.DataFrame(adata.obs["louvain"].unique()).reset_index()
    cell_type_df.columns = ["cell_type_idx", "louvain"]
    df_obs = adata.obs.merge(cell_type_df, on="louvain", how="left")
    df_obs.index = df_obs.index.astype(str)
    adata.obs = df_obs

    if not silent:
        print(adata)

    return adata