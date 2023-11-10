import warnings

warnings.warn(
    "dynamicdq has been moved to stdatamodels.dynamicdq, please use that instead",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from stdatamodels.dynamicdq import dynamic_mask
except ImportError as err:
    raise ImportError(
        "dynamicdq has been moved to stdatamodels.dynamicdq, please install stdatamodels"
    ) from err


__all__ = ["dynamic_mask"]
