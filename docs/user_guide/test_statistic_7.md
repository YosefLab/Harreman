# Test statistic 7: Do genes *a* and *b* interact when *a* is expressed by cell *i* and *b* by spatially nearby cells?

The results obtained using the cell type-agnostic approach described in Test statistics [1](test_statistic_1.md) and [2](test_statistic_2.md) can be visualized in space in a way such that each cell/spot is assigned a score aggregating the communication with neighboring cells/spots through a given gene pair $(a,b)$:

$$ H_{i} (ab) = \sum_{j}^{} w_{ij}(X_{ai}X_{bj} + X_{bi}X_{aj}) $$

If $a = b$:

$$ H_{i} (a) = \sum_{j}^{} w_{ij}X_{ai}X_{aj} $$

Statistical significance assessment is considered optional in this case, as it is possible to apply the above formula to just visualize the scores in space without filtering out non-significant values. However, an empirical approach has also been implemented for the purpose of avoiding considering interactions with low signal, and therefore visualizing only those cells or spots with significant scores. In this setting, if genes _a_ and _b_ are different, cell IDs in the counts matrix corresponding to gene _a_ (or _b_) are shuffled _M_ times ($M=1,000$ by default) and eventually, the most conservative p-values are selected. If gene $a=b$, though, the cell IDs in the counts matrix are shuffled. Then, the correlation values for each gene according to the equations above are computed in each iteration, respectively, and $p-value = \frac{x+1}{M+1}$ is used to calculate the p-value. These p-values are then adjusted using the Benjamini-Hochberg method.

Apart from using the empirical test, selecting the cells or spots based on the relative _H_ scores is also a good option, where threshold values such as 1 standard deviation above the mean are especially recommended.
