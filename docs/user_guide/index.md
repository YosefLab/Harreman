# User guide

## The Harreman algorithm for layered analysis of metabolic zonation and crosstalk in spatial data

Harreman provides a series of formulas to perform spatial correlation and characterize the metabolic state of tissue using spatial transcriptomics. At the coarsest level, Harreman partitions the tissue into modules of different metabolic functions based on enzyme co-expression. At the following stage, Harreman formulates hypotheses about which metabolites are exchanged across the tissue or within each spatial zone. This is achieved by computing an aggregate spatial correlation that takes into account all gene pairs associated with the import or export of a given metabolite. Moving to a finer resolution, Harreman can also infer which specific cell subsets participate in the exchange of distinct metabolic activities inside each zone. In this case, the aggregate spatial correlation is calculated by considering cell-type–restricted expression patterns of import–export gene pairs. Beyond these central functionalities, which are the main focus of this work, Harreman also provides a range of aggregate spatial correlation statistics designed to capture diverse interaction scenarios.

```{eval-rst}
.. card:: Level 1: Is gene *a* spatially autocorrelated?
    :link: level_1
    :link-type: doc
```
```{eval-rst}
.. card:: Level 2: Are genes *a* and *b* spatially co-localized?
    :link: level_2
    :link-type: doc
```
```{eval-rst}
.. card:: Level 3: Are genes *a* and *b* interacting with each other?
    :link: level_3
    :link-type: doc
```
```{eval-rst}
.. card:: Level 4: Is metabolite *m* spatially autocorrelated?
    :link: level_4
    :link-type: doc
```
```{eval-rst}
.. card:: Level 5: Do genes *a* and *b* interact when expressed by cell types *t* and *u*, respectively?
    :link: level_5
    :link-type: doc
```
```{eval-rst}
.. card:: Level 6: Is metabolite *m* exchanged by cell types *t* and *u*?
    :link: level_6
    :link-type: doc
```
```{eval-rst}
.. card:: Level 7: Do genes *a* and *b* interact when *a* is expressed by cell *i* and *b* by spatially nearby cells?
    :link: level_7
    :link-type: doc
```
```{eval-rst}
.. card:: Level 8: Is metabolite *m* exchanged by cell *i* and other spatially proximal cells?
    :link: level_8
    :link-type: doc
```
```{eval-rst}
.. card:: Level 9: Do genes *a* and *b* interact when *a* is expressed by cell *i* (that belongs to cell type *t*) and *b* by spatially nearby cells (that belong to cell type *u*)?
    :link: level_8
    :link-type: doc
```
```{eval-rst}
.. card:: Level 10: Is metabolite *m* exchanged by cell *i* (that belongs to cell type *t*) and other spatially proximal cells (that belong to cell type *u*)?
    :link: level_8
    :link-type: doc
```
