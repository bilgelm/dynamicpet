"""Sphinx configuration."""

project = "Dynamic PET"
author = "Murat Bilgel"
copyright = "2023, Murat Bilgel"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_nb",
]
autodoc_typehints = "description"
autosummary_generate = True
html_theme = "furo"
