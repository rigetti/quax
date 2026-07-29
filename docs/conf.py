"""Sphinx configuration for Quax documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

os.environ["JAX_ENABLE_X64"] = "1"

# -- Project information -----------------------------------------------------
project = "Quax"
copyright = "2026, Rigetti Computing"
author = "Bram Evert"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    "sphinxcontrib.bibtex",
]

# -- BibTeX configuration ----------------------------------------------------
bibtex_bibfiles = ["citations.bib"]
bibtex_default_style = "unsrt"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = "auto"
nbsphinx_allow_errors = False
nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    (
        "--InteractiveShellApp.exec_lines="
        """['import plotly.io as pio', 'pio.renderers.default = "sphinx_gallery"']"""
    ),
]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "Quax"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

# Include Plotly JS so interactive plots from notebooks render in the docs
html_js_files = [
    # Adding the 'defer' attribute helps it play nicely with Furo's DOM loading
    ("https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js", {"defer": "defer"}),
    ("https://cdn.plot.ly/plotly-2.35.2.min.js", {"defer": "defer"}),
]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#1976D2",
        "color-brand-content": "#1976D2",
    },
    "dark_css_variables": {
        "color-brand-primary": "#42A5F5",
        "color-brand-content": "#42A5F5",
    },
}

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Autosummary settings
autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
