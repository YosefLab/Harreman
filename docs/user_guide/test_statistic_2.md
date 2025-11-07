# Test statistic 2: Are genes *a* and *b* spatially co-localized (or interacting with each other)?

Once metabolic genes with a relevant spatial gene expression pattern have been selected, pairwise correlation can be computed to eventually group genes into modules and define tissue zones. Here we made use of the second formula of the Hotspot algorithm (DeTomaso and Yosef, _Cell systems_, 2021), which is defined as follows:

$$ H_{ab} = \sum_{i}^{}\sum_{j}^{} w_{ij} \left(X_{ai}X_{bj} + X_{bi}X_{aj}\right) $$

where _a_ and _b_ are two different genes expressed by cells _i_ and _j_, respectively, and _X_ refers to the gene expression matrix of dimension genes x cells.

The weight $w_{ij}$ represents communication strength between neighboring cells, and it is defined in the same way as in [Test statistic 1](test_statistic_1.md).

This statistic is also relevant to address the question of which interactions happen in the defined tissue regions. To infer metabolite exchange (or ligand-receptor interaction) events present in the tissue without adding the cell type constraint, a cell-type-agnostic approach has also been implemented. This analysis, despite not making use of any novel formulas, is based on gene pairs defined in HarremanDB and/or CellChatDB (Jin et al., _Nature protocols_, 2024), therefore restricting the communication analysis to already defined gene pairs that are known to interact with each other. As gene pairs can be made up of either different or the same genes, the formula needs to be adapted to each case. For the former case, the pairwise correlation formula above is used, while for the latter, that is, if $a = b$, the spatial autocorrelation formula defined in [Test statistic 1](test_statistic_1.md) is used.

For significance testing, both an empirical and a theoretical test have been implemented. The latter corresponds to the already existing theoretical test introduced in the Hotspot method (DeTomaso and Yosef, _Cell systems_, 2021). In this case, instead of considering a null model that assumes the expression values of genes _a_ and _b_ are independent, which significantly underestimates the variance of $H_{ab}$ if at least one gene has high autocorrelation (which is required to select these genes), a conditionally independent null hypothesis is tested. Here, we test how extreme $H_{ab}$ is compared with independent values of gene _b_ given the observed value of gene _a_, that is, $P(H_{ab}|a)$, and vice versa, that is, $P(H_{ab}|b)$. Eventually, we conservatively retain the least-significant result.
After going through equations defined in [Test statistic 1](test_statistic_1.md) adapted for $H_{ab} = \sum_{i}^{}\sum_{j}^{} w_{ij} \left(X_{ai}X_{bj} + X_{bi}X_{aj}\right)$, and conditioning on gene _a_, the second moment of _H_ is expressed as follows:

$$ E[\hat{H}_{ab}^2]= \sum_{i}^{}\left(\sum_{j \in N(i)}^{} w_{ij}\hat{X}_{bj}\right)^2 $$

To assess communication significance, the statistic is converted into a Z-score using the equation below, and a significance value is obtained for every gene by comparing the Z values to the normal distribution:

$$ \hat{Z}_{ab} = \frac{\hat{H}_{ab} - E[\hat{H}_{ab}]}{var(\hat{H}_{ab})^\frac{1}{2}} = \frac{\sum_{i}^{}\sum_{j}^{} w_{ij} \left(\hat{X}_{ai}\hat{X}_{bj} + \hat{X}_{bi}\hat{X}_{aj}\right)}{\left(\sum_{i}^{}\sum_{j}^{} w_{ij}^2\right)^\frac{1}{2}} $$

P-values are then adjusted using the Benjamini-Hochberg procedure.

In the empirical test setting, cell IDs in the counts matrix corresponding to gene _a_ (or _b_) are shuffled _M_ times ($M=1,000$ by default). Then, the correlation values for each gene according to $H_{ab} = \sum_{i}^{}\sum_{j}^{} w_{ij} \left(X_{ai}X_{bj} + X_{bi}X_{aj}\right)$ are computed in each iteration, and $p-value = \frac{x+1}{M+1}$ is used to calculate the p-value, where _x_ represents the number of permuted _H_ values that are higher than the observed one and _M_ is the total number of permutations. Finally, similar to the parametric test, the most conservative p-values are considered. These p-values are then adjusted using the Benjamini-Hochberg method.
