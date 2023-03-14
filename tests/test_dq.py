import pytest
import importlib

from contextlib import nullcontext

try:
    import stdatamodels  # noqa: F401
except ImportError:
    HAS_STDATAMODELS = False
else:
    HAS_STDATAMODELS = True


@pytest.mark.parametrize("name", ("dqflags", "dynamicdq", "basic_utils"))
def test_deprecation(name):
    error = (
        nullcontext()
        if HAS_STDATAMODELS
        else pytest.raises(
            ImportError, match=f"{name} has been moved to stdatamodels.{name},.*"
        )
    )

    with pytest.warns(
        DeprecationWarning, match=f"{name} has been moved to stdatamodels.{name},.*"
    ), error:
        importlib.import_module(f"stcal.{name}")
