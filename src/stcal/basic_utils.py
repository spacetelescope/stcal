import warnings

warnings.warn(
    "basic_utils has been moved to stdatamodels.basic_utils, please use that instead",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from stdatamodels.basic_utils import multiple_replace
except ImportError as err:
    msg = "basic_utils has been moved to stdatamodels.basic_utils, please install stdatamodels"
    raise ImportError(msg) from err


__all__ = ["multiple_replace"]
