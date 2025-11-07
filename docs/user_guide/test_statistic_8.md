# Test statistic 8: Is metabolite *m* exchanged by cell *i* and other spatially proximal cells?

Visualizing the individual cell/spot contributions of the results obtained using the metabolite-based approach described in [Test statistic 3](test_statistic_3.md) is also useful. For this, the equation below can be used for every metabolite (or ligand-receptor pathway), where the scores of the gene pairs (defined in [Test statistic 7](test_statistic_7.md)) that are associated with a given metabolite are summed up:

$$ H_{i}(m) = \sum_{a,b \in m}^{} H_{i} (ab) $$

To test for statistical significance, an empirical test has been implemented. Here, the equation above is computed in every shuffling iteration. This is done separately for genes _a_ and _b_ because, when _a_ and _b_ are different, the cell IDs corresponding to the counts matrices of both gene _a_ and _b_ are shuffled, giving rise to $H_i(m)(a)$ and $H_i(m)(b)$. However, if gene $a=b$, then the same output is considered for both $H_i(m)(a)$ and $H_i(m)(b)$. Then, two sets of p-values are obtained through the $p-value = \frac{x+1}{M+1}$ equation, and eventually, the most conservative p-values are selected. P-values are also adjusted using the Benjamini-Hochberg procedure.

Apart from using the empirical test, selecting the cells or spots based on the relative _H_ scores is also a good option, where threshold values such as 1 standard deviation above the mean are especially recommended.
