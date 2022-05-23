import importlib
import sys
from datetime import datetime
from pathlib import Path

import stsci_rtd_theme
import toml as toml


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
conf = toml.load(str(REPO_ROOT / 'pyproject.toml'))
setup_metadata = conf['project']

project = setup_metadata["name"]
author = setup_metadata["authors"][0]['name']
copyright = f"{datetime.now().year}, {author}"

package = importlib.import_module(setup_metadata["name"])
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