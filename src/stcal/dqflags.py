"""
Implementation
--------------

The flags are implemented as "bit flags": Each flag is assigned a bit position
in a byte, or multi-byte word, of memory. If that bit is set, the flag assigned
to that bit is interpreted as being set or active.

The data structure that stores bit flags is just the standard Python `int`,
which provides 32 bits. Bits of an integer are most easily referred to using
the formula `2**bit_number` where `bit_number` is the 0-index bit of interest.
"""
from astropy.nddata.bitmask import interpret_bit_flags as ap_interpret_bit_flags
from stcal.basic_utils import multiple_replace


def interpret_bit_flags(bit_flags, flip_bits=None, mnemonic_map=None):
    """Converts input bit flags to a single integer value (bit mask) or `None`.

    Wraps `astropy.nddate.bitmask.interpret_bit_flags`, allowing the
    bit mnemonics to be used in place of integers.

    Parameters
    ----------
    bit_flags : int, str, list, None
        See `astropy.nddate.bitmask.interpret_bit_flags`.
        Also allows strings using Roman mnemonics

    flip_bits : bool, None
        See `astropy.nddata.bitmask.interpret_bit_flags`.

    mnemonic_map : {str: int[,...]}
        Dictionary associating the mnemonic string to an integer value
        representing the set bit for that mnemonic.

    Returns
    -------
    bitmask : int or None
        Returns an integer bit mask formed from the input bit value or `None`
        if input ``bit_flags`` parameter is `None` or an empty string.
        If input string value was prepended with '~' (or ``flip_bits`` was set
        to `True`), then returned value will have its bits flipped
        (inverse mask).
    """
    if mnemonic_map is None:
        raise TypeError("`mnemonic_map` is a required argument")
    bit_flags_dm = bit_flags
    if isinstance(bit_flags, str):
        dm_flags = {
            key: str(val)
            for key, val in mnemonic_map.items()
        }
        bit_flags_dm = multiple_replace(bit_flags, dm_flags)

    return ap_interpret_bit_flags(bit_flags_dm, flip_bits=flip_bits)


def dqflags_to_mnemonics(dqflags, mnemonic_map):
    """Interpret value as bit flags and return the mnemonics

    Parameters
    ----------
    dqflags : int-like
        The value to interpret as DQ flags

    mnemonic_map: {str: int[,...]}
        Dictionary associating the mnemonic string to an integer value
        representing the set bit for that mnemonic.

    Returns
    -------
    mnemonics : {str[,...]}
        Set of mnemonics represented by the set bit flags

    Examples
    --------
    >>> pixel = {'GOOD':             0,      # No bits set, all is good
    ...          'DO_NOT_USE':       2**0,   # Bad pixel. Do not use
    ...          'SATURATED':        2**1,   # Pixel saturated during exposure
    ...          'JUMP_DET':         2**2,   # Jump detected during exposure
    ...          }

    >>> group = {'GOOD':       pixel['GOOD'],
    ...          'DO_NOT_USE': pixel['DO_NOT_USE'],
    ...          'SATURATED':  pixel['SATURATED'],
    ...          }

    >>> dqflags_to_mnemonics(1, pixel)
    {'DO_NOT_USE'}

    >>> dqflags_to_mnemonics(7, pixel)             #doctest: +SKIP
    {'JUMP_DET', 'DO_NOT_USE', 'SATURATED'}

    >>> dqflags_to_mnemonics(7, pixel) == {'JUMP_DET', 'DO_NOT_USE', 'SATURATED'}
    True

    >>> dqflags_to_mnemonics(1, mnemonic_map=pixel)
    {'DO_NOT_USE'}

    >>> dqflags_to_mnemonics(1, mnemonic_map=group)
    {'DO_NOT_USE'}
    """
    mnemonics = {
        mnemonic
        for mnemonic, value in mnemonic_map.items()
        if (dqflags & value)
    }
    return mnemonics
