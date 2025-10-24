# Level 10: Is metabolite *m* exchanged by cell *i* (that belongs to cell type *t*) and other spatially proximal cells (that belong to cell type *u*)?

In an analogous way to $H_{i \in C_t}^{t,u} (ab) = \sum_{j \in C_u}^{} w_{ij}X_{ai}X_{bj}$, the equation below represents the individual cell/spot metabolite (or LR pathway) values when cell type identity is considered:

$$ H_{i \in C_t}^{t,u} (m) = \sum_{a,b \in m}^{} H_{i \in C_t}^{t,u} (ab) $$

This equation is only used for visualization purposes, and no statistical test has been implemented.
