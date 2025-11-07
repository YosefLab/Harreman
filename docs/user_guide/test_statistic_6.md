# Level 6: Is metabolite *m* exchanged by cell types *t* and *u*?

Gene pairs can also be linked to metabolites by using the HarremanDB database (or to LR pathways by using CellChatDB). Therefore, as $H_{ab}^{t,u} = \sum_{i \in C_t}^{}\sum_{j \in C_u}^{} w_{ij}X_{ai}X_{bj}$ can be summed up for all gene pairs that exchange a given metabolite, we obtain the equation below:

$$ H_{m}^{t,u} = \sum_{a,b \in m}^{} H_{ab}^{t,u} $$

where _m_ is a given metabolite that is being exchanged by genes _a_ and _b_. In this setting, we would be assessing if metabolite _m_ is exchanged by cell types _t_ and _u_.

Significance testing in this case also depends on which of the three possible null hypotheses we are testing and, therefore, the shuffling strategy used. However, irrespective of the null hypothesis we want to test, the equation above is computed in each iteration. Eventually, p-values are computed ($p-value = \frac{x+1}{M+1}$) and adjusted using the Benjamini-Hochberg approach.
