import importlib
import sys
from datetime import datetime
from pathlib import Path

import stsci_rtd_theme

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib


def setup(app):
    try:
        app.add_css_file("stsci.css")
    except AttributeError:
        app.add_stylesheet("stsci.css")


REPO_ROOT = Path(__file__).parent.parent

# Modules that automodapi will document need to be available
# in the path:
sys.path.insert(0, str(REPO_ROOT / "src" / "stcal"))

# Read the package's `pyproject.toml` so that we can use relevant
# values here:
with (REPO_ROOT / "pyproject.toml").open("rb") as configuration_file:
    conf = tomllib.load(configuration_file)
setup_metadata = conf["project"]

project = setup_metadata["name"]
primary_author = setup_metadata["authors"][0]
author = f'{primary_author["name"]} <{primary_author["email"]}>'
copyright = f'{datetime.now().year}, {primary_author["name"]}'  # noqa: A001

package = importlib.import_module(project)
version = package.__version__.split("-", 1)[0]
release = package.__version__

# Configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("http://scipy.github.io/devdocs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "gwcs": ("https://gwcs.readthedocs.io/en/latest/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

extensions = [
    "pytest_doctestplus.sphinx.doctestplus",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.automodsumm",
    "sphinx_automodapi.autodoc_enhancements",
    "sphinx_automodapi.smart_resolver",
    "sphinx_asdf",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
numpydoc_show_class_members = False
autoclass_content = "both"

html_theme = "stsci_rtd_theme"
html_theme_options = {"collapse_navigation": True}
html_theme_path = [stsci_rtd_theme.get_html_theme_path()]
html_domain_indices = True
html_sidebars = {"**": ["globaltoc.html", "relations.html", "searchbox.html"]}
html_use_index = True

# Enable nitpicky mode - which ensures that all references in the docs resolve.
nitpicky = True
