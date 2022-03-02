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

        # Slice info
        self.start_row = None
        self.num_rows = None

        # One group ramp suppression for saturated ramps after 0th group.
        self.suppress_one_group_ramps = False
        self.one_groups = None

    def set_arrays(self, data, err, groupdq, pixeldq, int_times):
        """
        Set the arrays needed for ramp fitting.

        Parameter
        ---------
        data : ndarray
            4-D array containing the pixel information.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        err : ndarray
            4-D array containing the error information.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        groupdq : ndarray (uint16)
            4-D array containing the data quality flags.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        pixeldq : ndarray (uint32)
            4-D array containing the pixel data quality information.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        int_times : list
            Time information for each integration.
        """
        # Get arrays from the data model
        self.data = data
        self.err = err
        self.groupdq = groupdq
        self.pixeldq = pixeldq
        self.int_times = int_times

    def set_meta(self, name, frame_time, group_time, groupgap,
                 nframes, drop_frames1=None):
        """
        Set the metainformation needed for ramp fitting.

        Parameter
        ---------
        name : str
            The instrument name.

        frame_time : float32
            The time to read one frame.

        group_time : float32
            The time to read one group.

        groupgap : int
            The number of frames that are not included in the group average

        nframes : int
            The number of frames that are included in the group average

        drop_frames1 :
            The number of frames dropped at the beginning of every integration.
            May not be used in some pipelines, so is defaulted to NoneType.
        """

        # Get meta information
        self.instrument_name = name

        self.frame_time = frame_time
        self.group_time = group_time
        self.groupgap = groupgap
        self.nframes = nframes

        # May not be available for all pipelines, so is defaulted to NoneType.
        self.drop_frames1 = drop_frames1

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
