
import sys
from datetime import datetime
from pathlib import Path
from importlib.metadata import metadata

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))

# Project information
# try:
#     info = metadata("harreman")  # distribution name, lowercase
# except Exception:
#     info = "harreman"
info = metadata("Harreman")
project_name = info.get("Name", "Harreman")
author = info.get("Author") or "Oier Etxezarreta Arrastoa"
copyright = f"{datetime.now():%Y}, {author}."
version = info.get("Version", "0.1.0")
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
# repository_url = "https://github.com/YosefLab/Harreman"
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = version

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "YosefLab",  # Username
    "github_repo": project_name,  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "hoverxref.extension",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
bibtex_reference_style = "author_year"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ["css/custom.css"]

html_title = "Harreman"
html_logo = "_static/Harreman_logo.png"

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"

