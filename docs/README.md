# Writing and maintaining documentation

Documentation for `stcal` is written in [Sphinx reStructuredText (`.rst`)](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
in this `docs/` directory, and is hosted online athttps://stcal.readthedocs.io### Building documentation locally
ReadTheDocs will automatically build documentation for your branch when you push a commit to a pull request on this GitHub repository, and host a temporary build with a visual diff.
However, it is also good practice to build the docs locally if you are editing them, to reduce frustration from small errors.

To build the docs locally (assuming you have [set up your environment as described in `CONTRIBUTING.md`](../CONTRIBUTING.md#creating-a-development-environment)):

```shell
cd docs/
pip install ..[docs]
make clean
make html
```

The docs will build to `docs/_build/html/`.
Open `docs/_build/html/index.html` to view the pages in your browser.
