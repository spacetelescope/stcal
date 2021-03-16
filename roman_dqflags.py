def interpret_bit_flags(bit_flags, flip_bits=None):
    """Converts input bit flags to a single integer value (bit mask) or `None`.

    Wraps `astropy.nddate.bitmask.interpret_bit_flags`, allowing the Roman
    bit mnemonics to be used in place of integers.

    Parameters
    ----------
    bit_flags : int, str, list, None
        See `astropy.nddate.bitmask.interpret_bit_flags`.
        Also allows strings using Roman mnemonics

    flip_bits : bool, None
        See `astropy.nddata.bitmask.interpret_bit_flags`.

    Returns
    -------
    bitmask : int or None
        Returns an integer bit mask formed from the input bit value or `None`
        if input ``bit_flags`` parameter is `None` or an empty string.
        If input string value was prepended with '~' (or ``flip_bits`` was set
        to `True`), then returned value will have its bits flipped
        (inverse mask).

    Examples
    --------
    Using Roman mnemonics:
    TBD
    """
    bit_flags_dm = bit_flags
    if isinstance(bit_flags, str):
        dm_flags = {
            key: str(val)
            for key, val in pixel.items()
        }
        bit_flags_dm = multiple_replace(bit_flags, dm_flags)

    return ap_interpret_bit_flags(bit_flags_dm, flip_bits=flip_bits)


def dqflags_to_mnemonics(dqflags, mnemonic_map=pixel):
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
    >>> dqflags_to_mnemonics(1)
    {'DO_NOT_USE'}

    >>> dqflags_to_mnemonics(7)             #doctest: +SKIP
    {'JUMP_DET', 'DO_NOT_USE', 'SATURATED'}

    >>> dqflags_to_mnemonics(7) == {'JUMP_DET', 'DO_NOT_USE', 'SATURATED'}
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
