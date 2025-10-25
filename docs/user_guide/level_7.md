# Level 7: Do genes *a* and *b* interact when *a* is expressed by cell *i* and *b* by spatially nearby cells?

The results obtained using the cell type-agnostic approach described in [Level 3](level_3.md) can be visualized in space in a way such that each cell/spot is assigned a score aggregating the communication with neighboring cells/spots through a given gene pair $(a,b)$:

$$ H_{i} (ab) = \sum_{j}^{} w_{ij}(X_{ai}X_{bj} + X_{bi}X_{aj}) $$

If $a = b$:

$$ H_{i} (a) = \sum_{j}^{} w_{ij}X_{ai}X_{aj} $$

Statistical significance assessment is considered optional in this case, as it is possible to apply the above formula to just visualize the scores in space without filtering out non-significant values. However, a theoretical and an empirical approach have also been implemented for the purpose of avoiding considering interactions with low signal, and therefore visualizing only those cells or spots with significant scores. The second moments were computed using the same procedure as in Levels [1](level_1.md) and [2](level_2.md), giving rise to the equations below:

$$ E\left[{\hat{H}_i(ab)}^{2}\right]= \left(\sum_{j \in N(i)}^{} w_{ij}\hat{X}_{bj}\right)^2 $$

If $a = b$:

$$ E\left[{\hat{H}_i(a)}^{2}\right]= \sum_{j}^{} w_{ij}^2 $$

To assess communication significance, the statistic is converted into a Z-score using the equations below, and a significance value is obtained for every gene by comparing the Z values to the normal distribution:

$$ \hat{Z}_i(ab) = \frac{\hat{H}_i(ab) - E[\hat{H}_i(ab)]}{var(\hat{H}_i(ab))^\frac{1}{2}} = \frac{\sum_{j}^{} w_{ij}(X_{ai}X_{bj} + X_{bi}X_{aj})}{\left(\sum_{j \in N(i)}^{} w_{ij}\hat{X}_{bj}\right)^2} $$

If $a = b$:

$$ \hat{Z}_i(a) = \frac{\hat{H}_i(a) - E[\hat{H}_i(a)]}{var(\hat{H}_i(a))^\frac{1}{2}} = \frac{\sum_{j}^{} w_{ij}X_{ai}X_{aj}}{(\sum_{j}^{} w_{ij}^2)^\frac{1}{2}} $$

P-values are then adjusted using the Benjamini-Hochberg procedure.

In the empirical test setting, if genes _a_ and _b_ are different, cell IDs in the counts matrix corresponding to gene _a_ (or _b_) are shuffled _M_ times ($M=1,000$ by default) and eventually the most conservative p-values are selected. If gene $a=b$, though, the cell IDs in the counts matrix are shuffled. Then, the correlation values for each gene according to the equations below are computed in each iteration, respectively:

$$ \hat{Z}_i(ab) = \frac{\hat{H}_i(ab) - E[\hat{H}_i(ab)]}{var(\hat{H}_i(ab))^\frac{1}{2}} = \frac{\sum_{j}^{} w_{ij}(X_{ai}X_{bj} + X_{bi}X_{aj})}{\left(\sum_{j \in N(i)}^{} w_{ij}\hat{X}_{bj}\right)^2} $$

$$ \hat{Z}_i(a) = \frac{\hat{H}_i(a) - E[\hat{H}_i(a)]}{var(\hat{H}_i(a))^\frac{1}{2}} = \frac{\sum_{j}^{} w_{ij}X_{ai}X_{aj}}{(\sum_{j}^{} w_{ij}^2)^\frac{1}{2}} $$

To calculate the p-value, the equation below is used:

$$ p-value = \frac{x+1}{M+1} $$

These p-values are then adjusted using the Benjamini-Hochberg method.

Despite having implemented both the empirical and theoretical tests, using the empirical one is more appropriate, as theoretical p-values do not strongly correlate with the empirical ones. Further, selecting the cells or spots based on the relative _H_ scores is also a good option, where threshold values such as 1 standard deviation above the mean are especially recommended.
