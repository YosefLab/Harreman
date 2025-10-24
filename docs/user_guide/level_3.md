# Level 3: Are genes *a* and *b* interacting with each other?

After having defined metabolic zones in the tissue, a relevant question is which interactions happen in those tissue regions. To infer metabolite exchange (or LRI) events present in the tissue without adding the cell type constraint, a cell-type-agnostic approach has also been implemented. This analysis, despite not making use of any novel formulas, is based on gene pairs defined in HarremanDB and/or CellChatDB, therefore restricting the communication analysis to already defined gene pairs that are known to interact with each other. As gene pairs can be made up of either different or the same genes, the formula needs to be adapted to each case. For the former case, the pairwise Hotspot correlation formula is used, that is, $H_{ab} = \sum_{i}^{}\sum_{j}^{} w_{ij} \left(X_{ai}X_{bj} + X_{bi}X_{aj}\right)$, while for the latter, that is, if $a = b$, the spatial autocorrelation formula is used, that is, $H_{a} = \sum_{i}^{}\sum_{j}^{} w_{ij}X_{ai}X_{aj}$.

For genes composed of multiple subunits, we compute either an algebraic or a geometric mean as done in SpatialDM (Li et al., _Nature communications_, 2023). These formulas are applied before standardizing the gene expression counts:

$$ X_{ai} = \frac{\sum_{l \in S_l}^{} X_{a_li}}{|S_l|}; X_{bj} = \frac{\sum_{r \in S_r}^{} X_{b_rj}}{|S_r|} $$

$$ X_{ai} = \left( \prod_{l \in S_l}^{} X_{a_li} \right)^{1/L}; X_{bj} = \left( \prod_{r \in S_r}^{} X_{b_rj} \right)^{1/R} $$

where _l_ is a subunit for gene _a_ that belongs to the set of subunits $S_L$, and _r_ is a subunit for gene _b_ that belongs to the set of subunits $S_R$. $|S_l|$ and $|S_r|$ denote the number of subunits for genes _a_ and _b_, respectively. The geometric mean is a more stringent approach, as genes with at least one subunit with zero expression will lead to an inactive gene.

For significance testing using the parametric approach, the equations defined in the previous two subsections are used, depending on whether the gene pairs are made up of the same or different genes. For the former case, the equation below is used to compute the Z-scores for the theoretical test:

$$ \hat{Z}_a = \frac{\hat{H}_{a} - E[\hat{H}_{a}]}{var(\hat{H}_{a})^\frac{1}{2}} = \frac{\sum_{i}^{}\sum_{j}^{} w_{ij}\hat{X}_{ai}\hat{X}_{aj}}{\left(\sum_{i}^{}\sum_{j}^{} w_{ij}^2\right)^\frac{1}{2}} $$ 

while the equation below is used for the latter:

$$ \hat{Z}_{ab} = \frac{\hat{H}_{ab} - E[\hat{H}_{ab}]}{var(\hat{H}_{ab})^\frac{1}{2}} = \frac{\sum_{i}^{}\sum_{j}^{} w_{ij} \left(\hat{X}_{ai}\hat{X}_{bj} + \hat{X}_{bi}\hat{X}_{aj}\right)}{\left(\sum_{i}^{}\sum_{j}^{} w_{ij}^2\right)^\frac{1}{2}} $$

The approach to perform the empirical test is also identical to the autocorrelation and pairwise correlation cases, where if gene $a=b$, the cell IDs in the counts matrix are shuffled. However, when _a_ and _b_ are different, the cell IDs corresponding to the count matrices of both gene _a_ and _b_ are shuffled, and eventually the most conservative p-values are selected. P-values from both tests are also adjusted using the Benjamini-Hochberg procedure.
