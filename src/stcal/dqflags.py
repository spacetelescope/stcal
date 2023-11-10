import warnings

warnings.warn(
    "dqflags has been moved to stdatamodels.dqflags, please use that instead",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from stdatamodels.dqflags import (
        ap_interpret_bit_flags,
        dqflags_to_mnemonics,
        interpret_bit_flags,
        multiple_replace,
    )
except ImportError as err:
    raise ImportError("dqflags has been moved to stdatamodels.dqflags, please install stdatamodels") from err


__all__ = ["ap_interpret_bit_flags", "multiple_replace", "interpret_bit_flags", "dqflags_to_mnemonics"]
