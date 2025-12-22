# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

"""Sphinx configuration for DeepSuite documentation."""

import os
import sys

# -- Path setup ----------------------------------------------------------------
# Add the project directory to the path so autodoc can find modules
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DeepSuite"
author = "Anton Feldmann"
copyright = f"2025, {author}"

# Load version from pyproject.toml
try:
    import toml

    with open("../pyproject.toml") as f:
        pyproject = toml.load(f)
    release = pyproject["project"]["version"]
    version = ".".join(release.split(".")[:2])  # Major.Minor
except Exception:
    release = "1.0.4"
    version = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
]

# Mock heavy/optional imports to avoid autodoc failures in environments
# without these packages (e.g., macOS without torchvision wheels)
autodoc_mock_imports = [
    "torchvision",
    "albumentations",
    "audiomentations",
    "torchviz",
    "appt",
    "deepsuite.modules",
    "deepsuite.config.schema",
    "deepsuite.loss.custom",
    "deepsuite.tracker.kalman",
    "deepsuite.tracker.sort",
    "deepsuite.tracker.deepsort",
    "deepsuite.model.complex",
    "deepsuite.model.doa",
    "deepsuite.model.feature.resnet",
    "deepsuite.model.feature.efficientnet",
    "deepsuite.model.feature.mobile",
    "deepsuite.model.feature.darknet",
    "deepsuite.model.feature.wavenet",
    "deepsuite.model.feature.fpn",
    "deepsuite.model.detection.yolo",
    "deepsuite.model.detection.centernet",
    "deepsuite.model.detection.detection",
]

# Generate autosummary automatically
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Accept both .rst and .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "logo.png"
html_title = f"{project} {version}"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2962ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#448aff",
        "color-brand-content": "#448aff",
    },
}

# -- Autodoc settings ----------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_inherit_docstrings = False

# -- Napoleon settings ---------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- MyST (Markdown) settings --------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "tasklist",
]

# -- Intersphinx mapping -------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
