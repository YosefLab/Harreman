# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))

# Project info
project = 'Harreman'
copyright = '2025, Oier Etxezarreta Arrastoa'
author = 'Oier Etxezarreta Arrastoa'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # generate docs from docstrings
    'sphinx.ext.napoleon',   # Google/NumPy style docstrings
    'sphinx.ext.viewcode',   # links to source code
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
