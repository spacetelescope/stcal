from astropy import units as u

INDENT = "    "


class RampData:
    def __init__(self):
        """Creates an internal ramp fit class."""
        # Arrays from the data model
        self.data = None
        self.groupdq = None
        self.pixeldq = None
        self.average_dark_current = None

        # Needed for CHARGELOSS recomputation
        self.orig_gdq = None
        self.algorithm = None

        # Meta information
        self.instrument_name = None
        self.read_pattern = None
        self.rejection_threshold = None

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
        self.flags_chargeloss = None

        # ZEROFRAME
        self.zframe_mat = None
        self.zframe_locs = None
        self.zframe_cnt = 0
        self.zeroframe = None

        # Slice info
        self.start_row = None
        self.num_rows = None

        # One group ramp suppression for saturated ramps after 0th group.
        self.suppress_one_group_ramps = False

        self.one_groups_locs = None  # One good group locations.
        self.one_groups_time = None  # Time to use for one good group ramps.

        self.current_integ = -1

        self.debug = False

    def set_arrays(
        self, data, groupdq, pixeldq, average_dark_current, orig_gdq=None, zeroframe=None
    ):
        """
        Set the arrays needed for ramp fitting.

        Parameter
        ---------
        data : ndarray
            4-D array containing the pixel information.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        groupdq : ndarray (uint16)
            4-D array containing the data quality flags.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        pixeldq : ndarray (uint32)
            4-D array containing the pixel data quality information.  It has dimensions
            (nintegrations, ngroups, nrows, ncols)

        average_dark_current : ndarray (float32)
            2-D array containing the average dark current. It has
            dimensions (nrows, ncols)

        orig_gdq : ndarray
            4-D array containing a copy of the original group DQ array.  Since
            the group DQ array can be modified during ramp fitting, this keeps
            around the original group DQ flags passed to ramp fitting.
        """
        # Get arrays from the data model
        if isinstance(data, u.Quantity):
            self.data = data.value
        else:
            self.data = data
        self.groupdq = groupdq
        self.pixeldq = pixeldq
        self.average_dark_current = average_dark_current

        # May be None for some use cases
        self.orig_gdq = orig_gdq
        self.zeroframe = zeroframe

    def set_meta(self, name, frame_time, group_time, groupgap, nframes, drop_frames1=None):
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
            May not be used in some pipelines, so is defaulted to None.
        """
        # Get meta information
        self.instrument_name = name

        self.frame_time = frame_time
        self.group_time = group_time
        self.groupgap = groupgap
        self.nframes = nframes

        # May not be available for all pipelines, so is defaulted to None.
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
        if self.algorithm is not None and self.algorithm.upper() == "OLS_C":
            self.flags_chargeloss = dqflags["CHARGELOSS"]
