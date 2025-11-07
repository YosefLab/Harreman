# Level 5: Do genes *a* and *b* interact when expressed by cell types *t* and *u*, respectively?

To go further and identify the cell types that exchange the most relevant metabolites (or LR pathways) in the metabolic regions of interest, a cell type-aware approach has also been developed. The mathematical representation of this test statistic, which is used to quantify the communication strength between a given pair of cell types, is as follows:

$$ H_{ab}^{t,u} = \sum_{i \in C_t}^{}\sum_{j \in C_u}^{} w_{ij}X_{ai}X_{bj} $$

where _i_ and _j_, in addition to being two different cells, belong to different cell types _t_ and _u_. $C_t$ and $C_u$ refer to the set of cells that belong to cell types _t_ and _u_, respectively. Further, _a_ and _b_ are two different genes expressed by cells _i_ and _j_, respectively, and _X_ refers to the gene expression matrix of dimension genes x cells.
Further, the expression of the same gene ($a = b$) between two different cells must also be considered, i.e., when inferring metabolic crosstalk between cells that express the same transporter.

The weight $w_{ij}$ represents communication strength between neighboring cells, and it is defined in the same way as in [Level 1](level_1.md).

Weights are assigned using a spatial proximity graph, such that $w_{ij}$ is only non-zero if cells _i_ and _j_ are neighbors and there are no self-edges. This last statement is not true when dealing with deconvoluted spot-based spatial data, where interactions between different cell types present in the same spot could be considered. In that case, though, each cell type inferred in each spot using spatial deconvolution methods is treated as a separate node in the graph, where instead of assigning a distance equal to 0, the assigned distance between cell types within the same spot is $\frac{d}{2}$, with _d_ being the spot diameter. For this, DestVI (Lopez et al., _Nature biotechnology_, 2022) or cell2location (Kleshchevnikov et al., _Nature biotechnology_, 2022) can be used to estimate the cell-type abundance in each spot as well as the cell-type-specific gene expression values. As a result, the double summation can be re-expressed as a sum over edges _E_, which results in the following sparse graph:

$$ H_{ab}^{t,u} = \sum_{\substack{
(i, j) \in E \\
i \in C_t, j \in C_u
}}^{} w_{ij}X_{ai}X_{bj} $$

For genes composed of multiple subunits, we compute either an algebraic or a geometric mean as done in SpatialDM (Li et al. _Nature communications_, 2023):

$$ X_{ai} = \frac{\sum_{l \in S_l}^{} X_{a_li}}{|S_l|}; X_{bj} = \frac{\sum_{r \in S_r}^{} X_{b_rj}}{|S_r|} $$

$$ X_{ai} = \left( \prod_{l \in S_l}^{} X_{a_li} \right)^{1/L}; X_{bj} = \left( \prod_{r \in S_r}^{} X_{b_rj} \right)^{1/R} $$

To test significance and evaluate expectations for _H_, a null model is needed. For this, an empirical test has been implemented, where the shuffling procedure varies for each one of the 3 different null models:

-  (1) Given the spatial co-localization of cell types _t_ and _u_, which gene pairs are significantly co-expressed by cell types _t_ and _u_, respectively? The null hypothesis is as follows: the observed co-expression of gene pair $(a,b)$ across cell types _t_ and _u_ is no stronger than expected by chance, given the spatial co-localization of cell types _t_ and _u_. Therefore, gene pair expression counts within their respective cell types are shuffled.

-  (2) Given the spatial autocorrelation of a given gene pair $(a, b)$ regardless of cell type, which cell types explain the observed co-localization? The null hypothesis is as follows: the observed co-expression of genes _a_ and _b_ is not enriched in any specific cell type pair, that is, it is random with respect to which cell types express them. In this case, cell type labels are shuffled.

-  (3) Given a fixed cell type (e.g., stem cells), we test interactions with other cell types. The null hypothesis is as follows: the observed spatial co-expression between gene _a_ in a cell type of interest and gene _b_ in another cell type _u_ is no stronger than expected if gene _b_'s expression were random in cell type _u_. Here, we fix the expression of gene _a_ in cell type _t_ and shuffle the expression of gene _b_ in cell type _u_.

Then, the correlation values for each cell type pair and gene pair are computed according to the equation below in each iteration:

$$ H_{ab}^{t,u} = \sum_{\substack{
(i, j) \in E \\
i \in C_t, j \in C_u
}}^{} w_{ij}X_{ai}X_{bj} $$

and $p-value = \frac{x+1}{M+1}$ is used to calculate the p-value. P-values are finally adjusted using the Benjamini-Hochberg approach.
