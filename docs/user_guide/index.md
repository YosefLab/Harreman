# User guide

## The Harreman algorithm for layered analysis of metabolic zonation and crosstalk in spatial data

Harreman provides a series of formulas to perform spatial correlation and characterize the metabolic state of tissue using spatial transcriptomics. At the coarsest level, Harreman partitions the tissue into modules of different metabolic functions based on enzyme co-expression. At the following stage, Harreman formulates hypotheses about which metabolites are exchanged across the tissue or within each spatial zone. This is achieved by computing an aggregate spatial correlation that takes into account all gene pairs associated with the import or export of a given metabolite. Moving to a finer resolution, Harreman can also infer which specific cell subsets participate in the exchange of distinct metabolic activities inside each zone. In this case, the aggregate spatial correlation is calculated by considering cell-type–restricted expression patterns of import–export gene pairs. Beyond these central functionalities, which are the main focus of this work, Harreman also provides a range of aggregate spatial correlation statistics designed to capture diverse interaction scenarios.

```{eval-rst}
.. card:: Level 1: Is gene \textit{a} spatially autocorrelated?
    :link: level_1
    :link-type: doc
```

