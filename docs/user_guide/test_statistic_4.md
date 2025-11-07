# Test statistic 4: Are metabolites $m_1$ and $m_2$ spatially co-localized?

Once we have identified which metabolites show a spatially informed pattern, we can go one step further and group metabolites with similar spatial abundance. For this, a modified version of the equation defined in [Test statistic 2](test_statistic_2.md) is used, which takes as input binarized metabolite scores per cell/spot instead of gene expression data:

$$ H_{{m_1}{m_2}} = \sum_{i }^{}\sum_{j }^{} w_{ij}\left(I_{i}(m_1)I_{j}(m_2) + I_{i}(m_2)I_{j}(m_1)\right) $$

$$
I_{i}(m) = 
\begin{cases}
    1, & \text{if } H_{i}(m) > \tau \\
    0, & \text{otherwise}
\end{cases}
$$

where $\tau$ is defined as 1 standard deviation above the mean, and $H_{i}(m)$ is defined in [Test statistic 8](test_statistic_8.md).

Similarly to the previous test statistics, the binarized metabolite scores are standardized before computing the pairwise correlation value $H_{{m_1}{m_2}}$. For this, the Bernoulli distribution is used, as it is the best option to model the data.

For significance testing, the theoretical and empirical tests implemented in [Test statistic 2](test_statistic_2.md) have been adapted to this statistic. In the former case, instead of considering a null model that assumes the binarized scores of metabolites $m_1$ and $m_2$ are independent, which significantly underestimates the variance of $H_{{m_1}{m_2}}$ if at least one metabolite has high autocorrelation (which is required to select these metabolites), a conditionally independent null hypothesis is tested. Here, we test how extreme $H_{{m_1}{m_2}}$ is compared with independent values of metabolite $m_2$ given the observed value of metabolite $m_1$ $(P(H_{{m_1}{m_2}}|m_1))$, and vice versa $(P(H_{{m_1}{m_2}}|m_2))$. Eventually, we conservatively retain the least-significant result.
After going through equations defined in [Test statistic 1](test_statistic_1.md) adapted for $H_{{m_1}{m_2}} = \sum_{i }^{}\sum_{j }^{} w_{ij}\left(I_{i}(m_1)I_{j}(m_2) + I_{i}(m_2)I_{j}(m_1)\right)$, and conditioning on metabolite $m_1$, the second moment of $H$ is expressed as follows:

$$ E[\hat{H}_{{m_1}{m_2}}^2]= \sum_{i}^{}\left(\sum_{j \in N(i)}^{} w_{ij}\hat{I}_{j}(m_2)\right)^2 $$

To assess communication significance, the statistic is converted into a Z-score using the equation below, and a significance value is obtained for every gene by comparing the Z values to the normal distribution:

$$ \hat{Z}_{{m_1}{m_2}} = \frac{\hat{H}_{{m_1}{m_2}} - E[\hat{H}_{{m_1}{m_2}}]}{var(\hat{H}_{{m_1}{m_2}})^\frac{1}{2}} = \frac{\sum_{i}^{}\sum_{j}^{} w_{ij} \left(\hat{I}_{i}(m_1)\hat{I}_{j}(m_2) + \hat{I}_{i}(m_2)\hat{I}_{j}(m_1)\right)}{\left(\sum_{i}^{}\left(\sum_{j \in N(i)}^{} w_{ij}\hat{I}_{j}(m_2)\right)^2\right)^\frac{1}{2}} $$

P-values are then adjusted using the Benjamini-Hochberg procedure.

In the empirical test setting, cell IDs in the counts matrix corresponding to metabolite $m_1$ (or $m_2$) are shuffled _M_ times ($M=1,000$ by default). Then, the correlation values for each metabolite according to $H_{{m_1}{m_2}} = \sum_{i }^{}\sum_{j }^{} w_{ij}\left(I_{i}(m_1)I_{j}(m_2) + I_{i}(m_2)I_{j}(m_1)\right)$ are computed in each iteration, and $p-value = \frac{x+1}{M+1}$ is used to calculate the p-value, where _x_ represents the number of permuted _H_ values that are higher than the observed one and _M_ is the total number of permutations. Finally, similar to the parametric test, the most conservative p-values are considered. These p-values are then adjusted using the Benjamini-Hochberg method.
