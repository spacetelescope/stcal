import logging
import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def do_dqinit(input_model, mask_model, guider_list=None):
    """Perform the dq_init step on a datamodel

    Parameters
    ----------
    input_model : input science datamodel
        The science datamodel to be corrected

    mask_model : mask datamodel
        The mask model to use in the correction

    Returns
    -------
    output_model : science datamodel
        The dq corrected science datamodel
    """

    # Inflate empty DQ array, if necessary
    check_dimensions(input_model)

    # Create output model as copy of input
    output_model = input_model.copy()

    mask_array = mask_model.dq

    # Set model-specific data quality in output
    if input_model.meta.exposure.type in guider_list:
        dq = np.bitwise_or(input_model.dq, mask_array)
        output_model.dq = dq
    else:
        dq = np.bitwise_or(input_model.pixeldq, mask_array)
        output_model.pixeldq = dq

    output_model['meta']['cal_step']['dq_init'] = 'COMPLETE'

    return output_model


def check_dimensions(input_model):
    """Check that the input model pixeldq attribute has the same dimensions as
    the image plane of the input model science data
    If it has dimensions (0,0), create an array of zeros with the same shape
    as the image plane of the input model.

    For the guiding modes, the GuiderRawModel has only a regular dq array (no pixeldq or groupdq)

    Parameters
    ----------
    input_model : Raw datamodel
        input datamodel

    Returns
    -------
    None
    """

    input_shape = input_model.data.shape

    # Temporarily commented until GuiderModels implemented
    # if isinstance(input_model, datamodels.GuiderRawModel):
    #     if input_model.dq.shape != input_shape[-2:]:
    #
    #         # If the shape is different, then the mask model should have
    #         # a shape of (0,0).
    #         # If that's the case, create the array
    #         if input_model.dq.shape == (0, 0):
    #             input_model.dq = np.zeros((input_shape[-2:])).astype('uint32')
    #         else:
    #             log.error("DQ array has the wrong shape: (%d, %d)" %
    #                       input_model.dq.shape)
    #
    # else:   # RampModel
    if True:
        if input_model.pixeldq.shape != input_shape[-2:]:

            # If the shape is different, then the mask model should have
            # a shape of (0,0).
            # If that's the case, create the array
            if input_model.pixeldq.shape == (0, 0):
                input_model.pixeldq = \
                    np.zeros((input_shape[-2:])).astype('uint32')
            else:
                log.error("Pixeldq array has the wrong shape: (%d, %d)" %
                          input_model.pixeldq.shape)

        # Perform the same check for the input model groupdq array
        if input_model.groupdq.shape != input_shape:
            if input_model.groupdq.shape == (0, 0, 0, 0):
                input_model.groupdq = np.zeros((input_shape)).astype('uint8')
            else:
                log.error("Groupdq array has wrong shape: (%d, %d, %d, %d)" %
                          input_model.groupdq.shape)
    return
