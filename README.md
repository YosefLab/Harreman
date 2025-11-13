# Harreman

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/YosefLab/Harreman/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/YosefLab/Harreman?logo=GitHub&color=yellow)](https://github.com/YosefLab/Harreman/stargazers)
[![PyPI](https://img.shields.io/pypi/v/harreman.svg)](https://pypi.org/project/harreman)
[![Harreman](https://github.com/YosefLab/Harreman/actions/workflows/test.yml/badge.svg)](https://github.com/YosefLab/Harreman/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/YosefLab/Harreman/branch/main/graph/badge.svg?token=KuSsL5q3l7)](https://codecov.io/gh/YosefLab/Harreman)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Downloads](https://pepy.tech/badge/harreman)](https://pepy.tech/project/harreman)
[![Docs](https://readthedocs.org/projects/harreman/badge/?version=latest)](https://harreman.readthedocs.io/en/latest/)

Harreman is an algorithm and open-source software for inference of metabolic exchanges in tissues using spatial transcriptomics. Harreman employs a series of spatial correlation statistics to enable multiscale insight: from stratification of tissues into regions with different metabolic characteristics, to inference of which metabolites are exchanged within each region, and to identification of the specific subsets of cells that are exchanging these metabolites (through import and export reactions).

![overview](docs/figs/Harreman_pipeline.png)

## Resources
- Tutorials, user guide, API reference, installation guide, and release notes are available in the [Documentation](https://harreman.readthedocs.io/en).

## Installation
We suggest using a package manager like `conda` or `mamba` to install the package.

```bash
conda create -n harreman-env python=3.12
conda activate harreman-env
pip install harreman
```

## Tutorials
- For tutorials on how to run the PyTorch version of Hotspot with Harreman, visit the [Hotspot tutorials](https://harreman.readthedocs.io/en/latest/tutorials/hotspot_tutorials.html).
- To get familiar with the Harreman pipeline on the case studies used in the manuscript, visit the [Harreman tutorials](https://harreman.readthedocs.io/en/latest/tutorials/harreman_tutorials.html).

## Test statistics
All test statistics available in the Harreman pipeline are described in the [User guide](https://harreman.readthedocs.io/en/latest/user_guide/index.html). Currently implemented test statistics are:
- [Test statistic 1](https://harreman.readthedocs.io/en/latest/user_guide/test_statistic_1.html): Is gene *a* spatially autocorrelated?
- [Test statistic 2](https://harreman.readthedocs.io/en/latest/user_guide/test_statistic_2.html): Are genes *a* and *b* spatially co-localized (or interacting with each other)?
- [Test statistic 3](https://harreman.readthedocs.io/en/latest/user_guide/test_statistic_3.html): Is metabolite *m* spatially autocorrelated?
- [Test statistic 4](https://harreman.readthedocs.io/en/latest/user_guide/test_statistic_4.html): Are metabolites $m_1$ and $m_2$ spatially co-localized?
- [Test statistic 5](https://harreman.readthedocs.io/en/latest/user_guide/test_statistic_5.html): Do genes *a* and *b* interact when expressed by cell types *t* and *u*, respectively?
- [Test statistic 6](https://harreman.readthedocs.io/en/latest/user_guide/test_statistic_6.html): Is metabolite *m* exchanged by cell types *t* and *u*?
- [Test statistic 7](https://harreman.readthedocs.io/en/latest/user_guide/test_statistic_7.html): Do genes *a* and *b* interact when *a* is expressed by cell *i* and *b* by spatially nearby cells?
- [Test statistic 8](https://harreman.readthedocs.io/en/latest/user_guide/test_statistic_8.html): Is metabolite *m* exchanged by cell *i* and other spatially proximal cells?
- [Test statistic 9](https://harreman.readthedocs.io/en/latest/user_guide/test_statistic_9.html): Do genes *a* and *b* interact when *a* is expressed by cell *i* (that belongs to cell type *t*) and *b* by spatially nearby cells (that belong to cell type *u*)?
- [Test statistic 10](https://harreman.readthedocs.io/en/latest/user_guide/test_statistic_10.html): Is metabolite *m* exchanged by cell *i* (that belongs to cell type *t*) and other spatially proximal cells (that belong to cell type *u*)?

## Reference
- **Metabolic zonation and characterization of tissue slices with spatial transcriptomics**.
  *bioRxiv, 2025*. https://doi.org/10.1101/2025.11.11.687271

