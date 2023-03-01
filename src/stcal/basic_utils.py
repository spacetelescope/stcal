import warnings

warnings.warn(
    "basic_utils has been moved to stdatamodels.basic_utils, please use that instead",
    DeprecationWarning,
)

try:
    from stdatamodels.basic_utils import multiple_replace
except ImportError:
    raise ImportError("basic_utils has been moved to stdatamodels.basic_utils, please install stdatamodels")


__all__ = [multiple_replace]
