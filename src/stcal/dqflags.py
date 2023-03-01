import warnings

warnings.warn(
    "dqflags has been moved to stdatamodels.dqflags, please use that instead",
    DeprecationWarning,
)

try:
    from stdatamodels.dqflags import (
        ap_interpret_bit_flags,
        multiple_replace,
        interpret_bit_flags,
        dqflags_to_mnemonics,
    )
except ImportError:
    raise ImportError("dqflags has been moved to stdatamodels.dqflags, please install stdatamodels")


__all__ = [ap_interpret_bit_flags, multiple_replace, interpret_bit_flags, dqflags_to_mnemonics]
