import warnings

warnings.warn(
    "basic_utils has been moved to stdatamodels.basic_utils, please use that instead",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from stdatamodels.basic_utils import multiple_replace
except ImportError as err:
    raise ImportError(
        "basic_utils has been moved to stdatamodels.basic_utils, please install stdatamodels"
    ) from err


__all__ = ["multiple_replace"]
