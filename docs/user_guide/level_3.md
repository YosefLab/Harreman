# Level 3: Are genes *a* and *b* interacting with each other?

After having defined metabolic zones in the tissue, a relevant question is which interactions happen in those tissue regions. To infer metabolite exchange (or LRI) events present in the tissue without adding the cell type constraint, a cell-type-agnostic approach has also been implemented. This analysis, despite not making use of any novel formulas, is based on gene pairs defined in HarremanDB and/or CellChatDB, therefore restricting the communication analysis to already defined gene pairs that are known to interact with each other. As gene pairs can be made up of either different or the same genes, the formula needs to be adapted to each case. For the former case, the pairwise Hotspot correlation formula is used, that is, $H_{ab} = \sum_{i}^{}\sum_{j}^{} w_{ij} \left(X_{ai}X_{bj} + X_{bi}X_{aj}\right)$, while for the latter, that is, if $a = b$, the spatial autocorrelation formula is used, that is, $H_{a} = \sum_{i}^{}\sum_{j}^{} w_{ij}X_{ai}X_{aj}$.

For genes composed of multiple subunits, we compute either an algebraic or a geometric mean as done in SpatialDM (Li et al., _Nature communications_, 2023). These formulas are applied before standardizing the gene expression counts:

$$ X_{ai} = \frac{\sum_{l \in S_l}^{} X_{a_li}}{|S_l|}; X_{bj} = \frac{\sum_{r \in S_r}^{} X_{b_rj}}{|S_r|} $$

$$ X_{ai} = \left( \prod_{l \in S_l}^{} X_{a_li} \right)^{1/L}; X_{bj} = \left( \prod_{r \in S_r}^{} X_{b_rj} \right)^{1/R} $$


