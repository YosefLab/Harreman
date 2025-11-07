# Test statistic 9: Do genes *a* and *b* interact when *a* is expressed by cell *i* (that belongs to cell type *t*) and *b* by spatially nearby cells (that belong to cell type *u*)?

In addition to visualizing the cell-type-agnostic gene pair scores (defined in [Test statistic 7](test_statistic_7.md)), the equation below represents the individual cell/spot values when cell-type identity is considered:

$$ H_{i \in C_t}^{t,u} (ab) = \sum_{j \in C_u}^{} w_{ij}X_{ai}X_{bj} $$

This equation is only used for visualization purposes, and no statistical test has been implemented.
