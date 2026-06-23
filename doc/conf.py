import os
import sys
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath(".."))

project = "DS2L-SOM"
copyright = "2026, Sandro Martens"
author = "Sandro Martens"
try:
    release = _pkg_version("ds2l-som")
except Exception:
    release = "0.3.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"DS2L-SOM {release}"

numpydoc_show_class_members = False
autosummary_generate = True

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "remove_config_comments": True,
}
