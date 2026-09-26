# Running tests for stcal

`stcal` has several test suites to ensure that functionality remains consistent and does not break when code changes.
In order for a change you make to the code to be accepted and merged, that change must pass existing tests, as well as any new tests you write that cover new functionality.

`stcal` uses `pytest` to define and run tests. To install `pytest` and other required testing tools to your [development environment](./CONTRIBUTING.md#creating-a-development-environment), install `stcal` with the `test` extra:

```shell
pip install -e .[test]
```

To run tests, simply run `pytest`:

```shell
pytest
```

`pytest` recursively searches the given directory (by default `.`) for any files with a name like `test_*.py`, and runs all functions it finds that have a name like `test_*`.

See the [`pytest` documentation](https://docs.pytest.org) for more instructions on using `pytest`.

> [!TIP]
> You can control where test results are written by adding `--basetemp=<PATH>` to your `pytest` command.
> `pytest` will wipe this directory clean for each test session, so make sure it is a scratch area.
