class RampData:
    def __init__(self):
        """
        Creates an internal ramp fit class.
        """
        # Arrays from the data model
        self.data = None
        self.err = None
        self.groupdq = None
        self.pixeldq = None
        self.int_times = None

        # Meta information
        self.instrument_name = None

        self.frame_time = None
        self.group_time = None
        self.groupgap = None
        self.nframes = None
        self.drop_frames1 = None

        # Data quality flags
        self.flags_do_not_use = None
        self.flags_jump_det = None
        self.flags_saturated = None
        self.flags_no_gain_val = None
        self.flags_unreliable_slope = None

    def set_arrays(self, model):
        """
        Set the arrays needed for ramp fitting.

        Sets the following arrays:
        data : 4-D array containing the pixel information.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        err : 4-D array containing the error information.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        groupdq :4-D array containing the data quality flags.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        pixeldq : 4-D array containing the pixel data quality information.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        int_times : list
            Time information for each integration (only JWST).


        Parameters
        ----------
        model : Data model
            JWST or Roman Ramp Model

        """
        # Get arrays from the data model
        self.data = model.data
        self.err = model.err
        self.groupdq = model.groupdq
        self.pixeldq = model.pixeldq
        if hasattr(model, 'int_times'):
            self.int_times = model.int_times

    def set_meta(self, model):
        """
        Set the meta information needed for ramp fitting.

        name: The instrument name.
        frame_time: The time to read one frame.
        group_time: The time to read one group.
        groupgap: The number of frames that are not included in the group average
        nframes: The number of frames that are included in the group average
        drop_frames1: The number of frames dropped at the beginning of every integration.
                      May not be used in some pipelines, so is defaulted to NoneType.

        Parameters
        ----------
        model : Data model
            JWST or ROman Ramp Model
        """
        # Get meta information
        self.instrument_name = model.meta.instrument.name

        self.frame_time = model.meta.exposure.frame_time
        self.group_time = model.meta.exposure.group_time
        self.groupgap = model.meta.exposure.groupgap
        self.nframes = model.meta.exposure.nframes

        # May not be available for all pipelines, so is defaulted to NoneType.
        if hasattr(model, 'drop_frames1'):
            self.drop_frames1 = model.exposure.drop_frames1

    def set_dqflags(self, dqflags):
        """
        Set the data quality flags needed for ramp fitting.

        Parameter
        ---------
        dqflags : dict
            A dictionary with specific key words needed for processing.
        """
        # Get data quality flags
        self.flags_do_not_use = dqflags["DO_NOT_USE"]
        self.flags_jump_det = dqflags["JUMP_DET"]
        self.flags_saturated = dqflags["SATURATED"]
        self.flags_no_gain_val = dqflags["NO_GAIN_VALUE"]
        self.flags_unreliable_slope = dqflags["UNRELIABLE_SLOPE"]
