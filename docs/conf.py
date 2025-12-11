# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

# -- Path setup ----------------------------------------------------------------
# Füge das Projektverzeichnis dem Pfad hinzu, damit autodoc Module findet
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aptt'
author = 'Anton Feldmann <anton.feldmann@gmail.com>'
copyright = f"2025, {author}"

# Version automatisch aus pyproject.toml laden (optional)
try:
    import toml
    with open("../pyproject.toml", "rb") as f:
        pyproject = toml.load(f)
    release = pyproject["tool"]["poetry"]["version"]
except Exception:
    release = "0.1.0"  # Fallback

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.autodoc_pydantic",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Akzeptiere .rst und .md
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]

# -- Napoleon-Einstellungen ----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- MyST (Markdown) Einstellungen ---------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
]

# -- Intersphinx Mapping -------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- autodoc-pydantic Einstellungen --------------------------------------------
autodoc_pydantic_model_show_field_summary = True
autodoc_pydantic_model_show_config_summary = True
autodoc_pydantic_model_undoc_field = True
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_field_list = False
autodoc_pydantic_field_signature_prefix = ""
autodoc_pydantic_field_doc_policy = "description"
