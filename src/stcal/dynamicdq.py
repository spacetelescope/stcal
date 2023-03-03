import numpy as np

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.NullHandler())


def dynamic_mask(input_model, mnemonic_map, inv=False):
    """
    Return a mask model given a mask with dynamic DQ flags.

    Dynamic flags define what each plane refers to using the DQ_DEF extension.

    Parameters
    ----------
    input_model : ``MaskModel``
        An instance of a Mask model defined in jwst or romancal.

    mnemonic_map : dict
        Dictionary of flag names and values.

    inv : bool
        If true, compress using the dq_def.  If false, decompress
        using the dq_def.

    Returns
    -------
    dqmask : ndarray
        A Numpy array
    """

    dq_table = input_model.dq_def
    # Get the DQ array and the flag definitions
    if (dq_table is not None and
        not np.isscalar(dq_table) and
        len(dq_table.shape) and
            len(dq_table)):
        #
        # Make an empty mask
        dqmask = np.zeros(input_model.dq.shape, dtype=input_model.dq.dtype)
        for record in dq_table:
            bitplane = record['VALUE']
            dqname = record['NAME'].strip()

            # Check that a flag in the 'dq_def' is a valid DQ flag.
            try:
                standard_bitvalue = mnemonic_map[dqname]
            except KeyError:
                log.warning('Keyword %s does not correspond to an existing '
                            'DQ mnemonic, so will be ignored' % (dqname))
                continue

            if not inv:
                # Decompress the DQ array using 'dq_def'.
                just_this_bit = np.bitwise_and(input_model.dq, bitplane)
                pixels = np.where(just_this_bit != 0)
                dqmask[pixels] = np.bitwise_or(dqmask[pixels], standard_bitvalue)
            else:
                # Compress the DQ array using 'dq_def'.
                just_this_bit = np.bitwise_and(input_model.dq, standard_bitvalue)
                pixels = np.where(just_this_bit != 0)
                dqmask[pixels] = np.bitwise_or(dqmask[pixels], bitplane)

    else:
        dqmask = input_model.dq

    return dqmask
