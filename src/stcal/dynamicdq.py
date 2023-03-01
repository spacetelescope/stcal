import warnings

warnings.warn(
    "dynamicdq has been moved to stdatamodels.dynamicdq, please use that instead",
    DeprecationWarning,
)

try:
    from stdatamodels.dynamicdq import dynamic_mask
except ImportError:
    raise ImportError("dynamicdq has been moved to stdatamodels.dynamicdq, please install stdatamodels")


__all__ = [dynamic_mask]
