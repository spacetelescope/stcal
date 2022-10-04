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

        # ZEROFRAME
        self.zframe_locs = None
        self.zframe_cnt = 0
        self.zeroframe = None

        # Slice info
        self.start_row = None
        self.num_rows = None

        # One group ramp suppression for saturated ramps after 0th group.
        self.suppress_one_group_ramps = False

        self.current_integ = -1

    def set_arrays(self, data, err, groupdq, pixeldq):
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
        """
        # Get arrays from the data model
        self.data = data
        self.err = err
        self.groupdq = groupdq
        self.pixeldq = pixeldq

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

    def dbg_print_types(self):
        # Arrays from the data model
        print("-" * 80)
        print("    Array Types:")
        print(f"data : {type(self.data)}")
        print(f"err : {type(self.err)}")
        print(f"groupdq : {type(self.groupdq)}")
        print(f"pixeldq : {type(self.pixeldq)}")

        self.dbg_print_meta()
        self.dbg_print_mp()
        self.dbg_print_zframe()
        self.dbg_print_1grp()

    def dbg_print_meta(self):
        # Meta information
        print("-" * 80)
        print("    Meta:")
        print(f"Instumet: {self.instrument_name}")

        print(f"Frame time : {self.frame_time}")
        print(f"Group time : {self.group_time}")
        print(f"Group Gap : {self.groupgap}")
        print(f"Nframes : {self.nframes}")
        print(f"Drop Frames : {self.drop_frames1}")

    def dbg_print_mp(self):
        # Multiprocessing
        print("-" * 80)
        print(f"Start row : {self.start_row}")
        print(f"Number of rows : {self.num_rows}")

    def dbg_print_zframe(self):
        # ZEROFRAME
        print("-" * 80)
        print("    ZEROFRAME:")
        print(f"zframe_locs : {type(self.zframe_locs)}")
        print(f"zeroframe : {type(self.zeroframe)}")

    def dbg_print_1grp(self):
        # One group ramp suppression for saturated ramps after 0th group.
        print("-" * 80)
        print("    One Group Suppression:")
        print(f"suppress_one_group_ramps : {type(self.suppress_one_group_ramps)}")

    def dbg_print_basic_info(self):
        # Arrays from the data model
        print("-" * 80)
        print(f"Shape : {self.data.shape}")
        print(f"data : {self.data}")
        print(f"err : {self.err}")
        print(f"groupdq : {self.groupdq}")
        print(f"pixeldq : {self.pixeldq}")

        self.dbg_print_meta()

    def dbg_print_pixel_info(self, row, col):
        print("-" * 80)
        print(f"    data :\n{self.data[:, :, row, col]}")
        print(f"    err :\n{self.err[:, :, row, col]}")
        print(f"    groupdq :\n{self.groupdq[:, :, row, col]}")
        print(f"    pixeldq :\n{self.pixeldq[row, col]}")
