"""Sphinx configuration."""
project = "Dynamic PET"
author = "Murat Bilgel"
copyright = "2022, Murat Bilgel"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
