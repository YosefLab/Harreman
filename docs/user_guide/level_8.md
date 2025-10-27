# Level 8: Is metabolite *m* exchanged by cell *i* and other spatially proximal cells?

Visualizing the individual cell/spot contributions of the results obtained using the metabolite-based approach described in [Level 4](level_4.md) is also useful. For this, the equation below can be used for every metabolite (or LR pathway), where the scores of the gene pairs (defined in $H_{i} (ab) = \sum_{j}^{} w_{ij}(X_{ai}X_{bj} + X_{bi}X_{aj})$ and $H_{i} (a) = \sum_{j}^{} w_{ij}X_{ai}X_{aj}$) that are associated with a given metabolite are summed up:

$$ H_{i}(m) = \sum_{a,b \in m}^{} H_{i} (ab) $$

To test for statistical significance using the theoretical test, a very similar approach to the one described in [Level 4](level_4.md) is used, where the second moments of $H_{i}(ab)$ are summed up for every metabolite depending on the gene pairs that are associated with it:

$$ E\left[{\hat{H}_i (m)}^{2}\right]= \sum_{a,b \in m}^{} E\left[{\hat{H}_i (ab)}^{2}\right] $$

Eventually, Z-scores can be computed using the following equation:

$$ \hat{Z}_{i}(m) = \frac{\hat{H}_{i}(m) - E[\hat{H}_{i}(m)]}{var(\hat{H}_{i}(m))^\frac{1}{2}} = \frac{\sum_{a,b \in m}^{} \hat{H}_{i} (ab)}{\left(\sum_{a,b \in m}^{}E[\hat{H}_{i}(ab)^2]\right)^\frac{1}{2}} $$

and p-values are obtained for every metabolite by comparing the Z values to the normal distribution. P-values are then adjusted using the Benjamini-Hochberg approach.

For significance testing using the empirical test, the following equation is computed in every shuffling iteration:

$$ \hat{Z}_i(a) = \frac{\hat{H}_i(a) - E[\hat{H}_i(a)]}{var(\hat{H}_i(a))^\frac{1}{2}} = \frac{\sum_{j}^{} w_{ij}X_{ai}X_{aj}}{(\sum_{j}^{} w_{ij}^2)^\frac{1}{2}} $$

This is done separately for genes _a_ and _b_ because, when _a_ and _b_ are different, the cell IDs corresponding to the counts matrices of both gene _a_ and _b_ are shuffled, giving rise to $H_i(m)(a)$ and $H_i(m)(b)$. However, if gene $a=b$, then the same output is considered for both $H_i(m)(a)$ and $H_i(m)(b)$. Then, two sets of p-values are obtained through the $p-value = \frac{x+1}{M+1}$ equation, and eventually the most conservative p-values are selected. P-values are also adjusted using the Benjamini-Hochberg procedure.

Despite having implemented both the empirical and theoretical tests, using the empirical one is more appropriate, as theoretical p-values do not strongly correlate with the empirical ones. Further, selecting the cells or spots based on the relative _H_ scores is also a good option, where threshold values such as 1 standard deviation above the mean are especially recommended.
