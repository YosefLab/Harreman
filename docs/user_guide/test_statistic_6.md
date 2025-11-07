# Test statistic 6: Is metabolite *m* exchanged by cell types *t* and *u*?

Gene pairs can also be linked to metabolites by using the HarremanDB database (or to ligand-receptor pathways by using CellChatDB (Jin et al., _Nature protocols_, 2024)). Therefore, as $H_{ab}^{t,u} = \sum_{i \in C_t}^{}\sum_{j \in C_u}^{} w_{ij}X_{ai}X_{bj}$ can be summed up for all gene pairs that exchange a given metabolite, we obtain the equation below:

$$ H_{m}^{t,u} = \sum_{a,b \in m}^{} H_{ab}^{t,u} $$

where _m_ is a given metabolite that is being exchanged by genes _a_ and _b_. In this setting, we would be assessing if metabolite _m_ is exchanged by cell types _t_ and _u_.

Significance testing in this case also depends on which of the three possible null hypotheses defined in [Test statistic 5](test_statistic_5.md) we are testing and, therefore, the shuffling strategy used. The null hypotheses in this case are defined as follows:

(1) The observed co-expression of all gene pairs associated with metabolite _m_ across cell types _t_ and _u_ is no stronger than expected by chance, given the spatial co-localization of cell types t and u. 

(2) The observed co-expression of metabolite _m_’s associated genes is not enriched in any specific cell type pair, that is, it is random with respect to which cell types express those genes.

(3) The observed spatial co-expression between the expression of metabolite _m_’s genes in a cell type of interest and the expression of those genes in another cell type _u_ is no stronger than expected if gene expression in cell type _u_ were random.

However, irrespective of the null hypothesis we want to test, the equation above is computed in each iteration. Eventually, p-values are computed ($p-value = \frac{x+1}{M+1}$) and adjusted using the Benjamini-Hochberg approach.
