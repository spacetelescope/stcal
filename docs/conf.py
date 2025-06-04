import importlib
import sys
from datetime import datetime
from pathlib import Path

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib


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
    "tweakwcs": ("https://tweakwcs.readthedocs.io/en/latest/", None),
    "drizzle": (
        "https://spacetelescope-drizzle.readthedocs.io/en/latest/",
        None
    ),
    "imagestats": ("https://stsciimagestats.readthedocs.io/en/latest/", None),
    "spherical_geometry": ("https://spherical-geometry.readthedocs.io/en/latest/", None),
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

html_theme = "sphinx_rtd_theme"
html_logo = "_static/stsci_pri_combo_mark_white.png"
html_theme_options = {
    "collapse_navigation": True,
}
html_domain_indices = True
html_sidebars = {"**": ["globaltoc.html", "relations.html", "searchbox.html"]}
html_use_index = True

# Enable nitpicky mode - which ensures that all references in the docs resolve.
nitpicky = False  # True does not work with enumerated parameter values

nitpick_ignore = [
    ("py:class", "optional"),
    ("py:class", "np.ndarray"),
    ("py:class", "stsci.imagestats.ImageStats"),  # intersphinx isn't working here
    ("py:class", "spherical_geometry.polygon.SphericalPolygon"), # intersphinx isn't working here
]

