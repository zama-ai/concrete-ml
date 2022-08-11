# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os

# import sys

# sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "Concrete-ML"
copyright = "2022, Zama"
author = "Zama"
description = "Zama Concrete-ML"
root_url = os.environ.get("DOC_ROOT_URL", "/concrete-ml")
root_url = root_url if root_url.endswith("/") else root_url + "/"

# The full version, including alpha/beta/rc tags
release = "0.4.0-rc0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinx.ext.napoleon",
    # Comment this extension when not at release time
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "SUMMARY.md",
    "advanced_examples/*.ipynb",
]

# Group member variables and methods separately (not alphabetically)
autodoc_member_order = "groupwise"

# -- Options for nbsphinx ----------------------------------------------------

nbsphinx_codecell_lexer = "ipython3"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# We may be back with sphinx_zama_theme, as soon as it provides something which is very close to
# GitBook look, or keep ReadsTheDocs theme e.g. if we want to push docs there
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Were options for sphinx_zama_theme theme
# html_theme_options = {
#     "github_url": "https://github.com/zama-ai/concrete-ml",
#     "twitter_url": "https://twitter.com/zama_fhe",
#     "icon_links": [
#         {
#             "name": "Discourse",
#             "url": "https://community.zama.ai/c/concrete-ml",
#             "icon": "fab fa-discourse",
#         }
#     ],
#     "navigation_depth": 2,
#     "collapse_navigation": True,
#     "google_analytics_id": "G-XRM93J9QBW",
# }

html_theme_options = {
    "analytics_id": "G-XRM93J9QBW",
    # 'analytics_anonymize_ip': False,
    # 'logo_only': False,
    # 'display_version': True,
    # 'prev_next_buttons_location': 'bottom',
    # 'style_external_links': False,
    # 'vcs_pageview_mode': '',
    # 'style_nav_header_background': 'white',
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": False,
    "navigation_depth": 0,
    "includehidden": True,
    "titles_only": True,
}
html_context = {
    "show_version": True,
    "author": author,
    "description": description,
    "language": "en",
    "versions_url": "#",
}
html_title = "%s Manual" % (project)
html_show_sphinx = False

# Uncomment for test
# html_extra_path = ["versions.json", "alert.html"]


def setup(app):
    html_init = f"const CURRENT_VERSION = {release!r};"
    html_init += f"const ROOT_URL = {root_url!r};"
    app.add_js_file(None, body=html_init, priority=100)
