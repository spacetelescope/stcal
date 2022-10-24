import importlib
import sys
from datetime import datetime
from pathlib import Path

import stsci_rtd_theme
import tomli


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
with open(REPO_ROOT / "pyproject.toml", "rb") as configuration_file:
    conf = tomli.load(configuration_file)
setup_metadata = conf['project']

project = setup_metadata["name"]
primary_author = setup_metadata["authors"][0]
author = f'{primary_author["name"]} <{primary_author["email"]}>'
copyright = f'{datetime.now().year}, {primary_author["name"]}'

package = importlib.import_module(project)
version = package.__version__.split("-", 1)[0]
release = package.__version__

extensions = [
    "sphinx_automodapi.automodapi",
    "numpydoc",
]

autosummary_generate = True
numpydoc_show_class_members = False
autoclass_content = "both"

html_theme = "stsci_rtd_theme"
html_theme_options = {
    "collapse_navigation": True
}
html_theme_path = [stsci_rtd_theme.get_html_theme_path()]
html_domain_indices = True
html_sidebars = {"**": ["globaltoc.html", "relations.html", "searchbox.html"]}
html_use_index = True

# Enable nitpicky mode - which ensures that all references in the docs resolve.
nitpicky = True
