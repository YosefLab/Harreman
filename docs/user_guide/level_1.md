# Level 1: Is gene *a* spatially autocorrelated?

When performing tissue zonation, having a limited set of informative genes is useful to constrain the analysis to spatially autocorrelated genes. For this, the equation defined in the Hotspot algorithm (DeTomaso and Yosef, _Cell systems_, 2021) is used on the set of metabolic genes used by the Compass algorithm (Wagner et al., _Cell_, 2021):

$$ H_{a} = \sum_{i}^{}\sum_{j}^{} w_{ij}X_{ai}X_{aj} $$

where _a_ is a gene expressed by cells _i_ and _j_, and _X_ refers to the gene expression matrix of dimension _genes_ x _cells_.

The weight $w_{ij}$ represents communication strength between neighboring cells. It is positive if _i_ and _j_ are neighbors and 0 otherwise, and the lower the distance in the similarity graph, meaning a more similar neighbor, the higher the value. To calculate them, a Gaussian kernel is used, which is defined by the following equation:

$$ \hat{w}_{ij} = e^{-d_{ij}^2/\sigma_{i}^2} $$

where $d_{ij}$ corresponds to the distance between two neighboring cells and $\sigma_{i}$ refers to the selected bandwidth for cell _i_, which by default is set to $K/3$, where _K_ represents the number of chosen neighbors in the K-nearest-neighbors (kNN) graph or the last neighbor at a distance smaller than _d_ micrometers, with the most appropriate values being between 50 and 200 micrometers to focus on local neighborhoods. An unweighted option is also available, where the value is 1 if two cells are neighbors and 0 otherwise.

Weights are also normalized across cells to sum 1 for each cell (this is only done in the weighted case).

$$ w_{ij} = \frac{\hat{w}_{ij}}{\sum_{j}^{}\hat{w}_{ij}} $$

Further, in cases where the dataset contains the spatial positions of two or more samples, to avoid connections between cells or spots that belong to different samples, a weight of 0 will be assigned if two given cells belong to different samples.

To test significance and evaluate expectations for _H_, that is, to assess if the obtained value is extreme compared to what we would expect by chance, a null model is needed. A parametric and a non-parametric approach have been implemented, the former having already been introduced in the Hotspot manuscript (DeTomaso and Yosef, _Cell systems_, 2021). The theoretical null assumes that expression values are drawn independently from some underlying distribution for which we can compute $E\left[X_{ai}\right]$ and $E\left[X_{ai}^{2}\right]$ for each cell. The null hypothesis here would be that the expression of gene _a_ is independently and identically distributed across cells, that is, that there is no spatial autocorrelation.

$$ E\left[H_{a}\right] = \sum_{i}^{}\sum_{j}^{} w_{ij}E\left[X_{ai}\right]E\left[X_{aj}\right] $$

$$ H_{a}^{2} = \sum_{(i,j) \in E}^{}\sum_{(k,l) \in E}^{} w_{ij}w_{kl}X_{ai}X_{aj}X_{ak}X_{al} $$

$$ E\left[H_{a}^{2}\right] = \sum_{(i,j) \in E}^{}\sum_{(k,l) \in E}^{} w_{ij}w_{kl}E\left[X_{ai}\right]E\left[X_{aj}\right]E\left[X_{ak}\right]E\left[X_{al}\right] $$

$$ var\left(H_{a}\right) = E\left[H_{a}^{2}\right] - E\left[H_{a}\right]^{2} $$

The expression values are standardized before computing the autocorrelation value _H_. For this, different statistical models can be used, where the negative binomial distribution is generally used to model single-cell and spatial data. However, in cases where the counts are very sparse, the Bernoulli distribution might be a better option. Moreover, the normal distribution has also been implemented to model other types of data that don't follow either of the previously mentioned models. The expression count standardization is performed as follows:

$$ \hat{X}_{ai} = \frac{X_{ai} - E\left[X_{ai}\right]}{var\left(X_{ai}\right)^\frac{1}{2}} $$

$$ \hat{H}_{a} = \sum_{i}^{}\sum_{j}^{} w_{ij}\hat{X}_{ai}\hat{X}_{aj} $$

Computing the null model is made simpler as the expectation of _H_ is 0:

$$E\left[\hat{X}_{ai}\right] = 0$$

$$E\left[\hat{X}_{ai}^{2}\right] = 1$$

$$ E\left[\hat{H}_{a}\right] = 0 $$

Therefore, the second moment of _H_ is as follows:

$$ E\left[\hat{H}_{a}^2\right]= \sum_{i}^{}\sum_{j}^{} w_{ij}^2 $$

To assess communication significance, like with the autocorrelation statistic, $H_{ab}$ is converted into a Z-score using the equation below, and a significance value is obtained for every gene pair by comparing the Z values to the normal distribution:

$$ \hat{Z}_a = \frac{\hat{H}_{a} - E[\hat{H}_{a}]}{var(\hat{H}_{a})^\frac{1}{2}} = \frac{\sum_{i}^{}\sum_{j}^{} w_{ij}\hat{X}_{ai}\hat{X}_{aj}}{\left(\sum_{i}^{}\sum_{j}^{} w_{ij}^2\right)^\frac{1}{2}} $$

Lastly, p-values are adjusted using the Benjamini-Hochberg procedure.

In the empirical test setting, either raw gene counts, log-normalized values, standardized gene expression values, or counts normalized in any other desired way can be used. Cell IDs from the counts matrix are shuffled _M_ times ($M=1,000$ by default), and the correlation values for each gene according to $H_{a} = \sum_{i}^{}\sum_{j}^{} w_{ij}X_{ai}X_{aj}$ are computed in each iteration. Then, the equation below is used to calculate the p-value, where _x_ represents the number of permuted _H_ values that are higher than the observed one and _M_ is the total number of permutations.

$$ p-value = \frac{x+1}{M+1} $$

Finally, p-values are adjusted using the Benjamini-Hochberg method.
