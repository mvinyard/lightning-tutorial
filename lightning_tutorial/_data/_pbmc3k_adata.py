
__module_name__ = "_pbmc3k_adata.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# import packages: -------------------------------------------------------
import scanpy as sc
import pandas as pd
import vinplots


# supporting functions: --------------------------------------------------
def _plot_adata_groupby(adata, groupby="louvain", use_key="X_umap", **kwargs):

    fig, axes = vinplots.quick_plot(
        nplots=1, ncols=1, wspace=0.1, rm_ticks=True, **kwargs
    )
    fig.modify_spines(ax="all", spines_to_delete=["left", "bottom"])
    cp = adata.uns["color_palette"]
    for group, group_df in adata.obs.groupby(groupby):
        xu = adata[group_df.index].obsm[use_key]
        axes[0].scatter(xu[:, 0], xu[:, 1], label=group, color=cp[group], alpha=0.65)
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")

    axes[0].legend(loc=(1, 0.4), edgecolor="w")
    axes[0].set_title("10x PBMCs (~3k cells)")

# primary function: ------------------------------------------------------
def _pbmc3k_adata(plot=True, silent=False):

    """Use the scanpy API to load the pbmc3k dataset and format the louvain/celltype labels as integers."""

    adata = sc.datasets.pbmc3k_processed()

    # format louvain (cell type) labels as integers
    cell_type_df = pd.DataFrame(adata.obs["louvain"].unique()).reset_index()
    cell_type_df.columns = ["cell_type_idx", "louvain"]
    df_obs = adata.obs.merge(cell_type_df, on="louvain", how="left")
    df_obs.index = df_obs.index.astype(str)
    adata.obs = df_obs
    
    adata.uns['color_palette'] = vinplots.colors.pbmc3k

    if not silent:
        print(adata)
        
    if plot:
        _plot_adata_groupby(adata)

    return adata