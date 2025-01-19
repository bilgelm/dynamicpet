"""Sphinx configuration."""

# ruff: noqa: INP001

project = "Dynamic PET"
author = "Murat Bilgel"
copyright = "2023, Murat Bilgel"  # noqa: A001
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_nb",
]
autodoc_typehints = "description"
autosummary_generate = True
html_theme = "furo"
nb_execution_timeout = -1
