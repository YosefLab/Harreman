# Level 4: Is metabolite *m* spatially autocorrelated?

Apart from getting interaction event information at the gene pair level, those results can be integrated to study one additional layer of intracellular interactions. As each gene pair used to compute _H_ is associated with a given metabolite (HarremanDB) or LR pathway (CellChatDB), the above formulas can be summed over all gene pairs that exchange a given metabolite (or interact through a given LR pathway):

$$ H_{m} = \sum_{a,b \in m}^{}H_{ab} $$

where _m_ is a given metabolite that is being exchanged by genes _a_ and _b_ (or $a = b$). In this setting, we would be assessing the spatial autocorrelation of metabolite _m_.

To test for statistical significance using the theoretical test, the same procedure is applied on the second moments of _H_, giving rise to the equation below:

$$ E[\hat{H}_{m}^2]= \sum_{a,b \in m}^{}E[\hat{H}_{ab}^2] $$

Eventually, Z-scores can be computed using the following equation:

$$ \hat{Z}_{m} = \frac{\hat{H}_{m} - E[\hat{H}_{m}]}{var(\hat{H}_{m})^\frac{1}{2}} = \frac{\sum_{a,b \in m}^{}\hat{H}_{ab}}{\left(\sum_{a,b \in m}^{}E[\hat{H}_{ab}^2]\right)^\frac{1}{2}} $$

and p-values are obtained for every metabolite by comparing the Z values to the normal distribution. P-values are then adjusted using the Benjamini-Hochberg approach.

For significance testing using the empirical test, the equation below is computed in every shuffling iteration:

$$ E[\hat{H}_{m}^2]= \sum_{a,b \in m}^{}E[\hat{H}_{ab}^2] $$

This is done separately for genes _a_ and _b_ because, when _a_ and _b_ are different, the cell IDs corresponding to the counts matrices of both gene _a_ and _b_ are shuffled, giving rise to $H_m (a)$ and $H_m (b)$. However, if gene $a=b$, then the same output is considered for both $H_m (a)$ and $H_m (b)$. Then, two sets of p-values are obtained through $p-value = \frac{x+1}{M+1}$, and eventually the most conservative p-values are selected. P-values are also adjusted using the Benjamini-Hochberg procedure.
