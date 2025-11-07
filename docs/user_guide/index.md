# User guide

Harreman provides a series of formulas to perform spatial correlation and characterize the metabolic state of tissue using spatial transcriptomics. At the coarsest level, Harreman partitions the tissue into modules of different metabolic functions based on enzyme co-expression. At the following stage, Harreman formulates hypotheses about which metabolites are exchanged across the tissue or within each spatial zone. This is achieved by computing an aggregate spatial correlation that takes into account all gene pairs associated with the import or export of a given metabolite. Spatially co-localized metabolites can also be grouped together. Moving to a finer resolution, Harreman can also infer which specific cell subsets participate in the exchange of distinct metabolic activities inside each zone. In this case, the aggregate spatial correlation is calculated by considering cell-type–restricted expression patterns of import–export gene pairs. Beyond these central functionalities, which are the main focus of this work, Harreman also provides a range of aggregate spatial correlation statistics designed to capture diverse interaction scenarios.

In all the equations described in the sections below, for proteins composed of multiple subunits, we compute either an algebraic or a geometric mean of the expression values of the corresponding genes as done in SpatialDM (Li et al., _Nature communications_, 2023):

$$ X_{ai} = \frac{\sum_{l \in S_l}^{} X_{a_li}}{|S_l|}; X_{bj} = \frac{\sum_{r \in S_r}^{} X_{b_rj}}{|S_r|} $$

$$ X_{ai} = \left( \prod_{l \in S_l}^{} X_{a_li} \right)^{1/L}; X_{bj} = \left( \prod_{r \in S_r}^{} X_{b_rj} \right)^{1/R} $$

where _l_ is a subunit for the protein encoded by gene _a_ that belongs to the set of subunits $S_L$, and _r_ is a subunit for the protein encoded by gene _b_ that belongs to the set of subunits $S_R$. $|S_l|$ and $|S_r|$ denote the number of subunits for proteins encoded by genes _a_ and _b_, respectively.

```{eval-rst}
.. card:: Test statistic 1: Is gene *a* spatially autocorrelated?
    :link: test_statistic_1
    :link-type: doc
```
```{eval-rst}
.. card:: Test statistic 2: Are genes *a* and *b* spatially co-localized (or interacting with each other)?
    :link: test_statistic_2
    :link-type: doc
```
```{eval-rst}
.. card:: Test statistic 3: Is metabolite *m* spatially autocorrelated?
    :link: test_statistic_3
    :link-type: doc
```
```{eval-rst}
.. card:: Test statistic 4: Are metabolites *m_1* and *m_2* spatially co-localized?
    :link: test_statistic_4
    :link-type: doc
```
```{eval-rst}
.. card:: Test statistic 5: Do genes *a* and *b* interact when expressed by cell types *t* and *u*, respectively?
    :link: test_statistic_5
    :link-type: doc
```
```{eval-rst}
.. card:: Test statistic 6: Is metabolite *m* exchanged by cell types *t* and *u*?
    :link: test_statistic_6
    :link-type: doc
```
```{eval-rst}
.. card:: Test statistic 7: Do genes *a* and *b* interact when *a* is expressed by cell *i* and *b* by spatially nearby cells?
    :link: test_statistic_7
    :link-type: doc
```
```{eval-rst}
.. card:: Test statistic 8: Is metabolite *m* exchanged by cell *i* and other spatially proximal cells?
    :link: test_statistic_8
    :link-type: doc
```
```{eval-rst}
.. card:: Test statistic 9: Do genes *a* and *b* interact when *a* is expressed by cell *i* (that belongs to cell type *t*) and *b* by spatially nearby cells (that belong to cell type *u*)?
    :link: test_statistic_9
    :link-type: doc
```
```{eval-rst}
.. card:: Test statistic 10: Is metabolite *m* exchanged by cell *i* (that belongs to cell type *t*) and other spatially proximal cells (that belong to cell type *u*)?
    :link: test_statistic_10
    :link-type: doc
```
