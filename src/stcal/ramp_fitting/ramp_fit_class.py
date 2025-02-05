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

        # C code debugging switch.
        self.run_c_code = False

        self.one_groups_locs = None  # One good group locations.
        self.one_groups_time = None  # Time to use for one good group ramps.

        self.current_integ = -1

        self.debug = False

    def set_arrays(self, data, groupdq, pixeldq, average_dark_current, orig_gdq=None):
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
        self.data = data
        self.groupdq = groupdq
        self.pixeldq = pixeldq
        self.average_dark_current = average_dark_current

        self.orig_gdq = orig_gdq

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
        if self.algorithm is not None and self.algorithm.upper() == "OLS_C":
            self.flags_chargeloss = dqflags["CHARGELOSS"]

    def dbg_print_types(self):
        # Arrays from the data model
        print("-" * 80)
        print("    Array Types:")
        print(f"data : {type(self.data)}")
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
        print(f"Instrument: {self.instrument_name}")

        print(f"Frame time : {self.frame_time}")
        print(f"Group time : {self.group_time}")
        print(f"Group Gap : {self.groupgap}")
        print(f"Nframes : {self.nframes}")
        print(f"Drop Frames : {self.drop_frames1}")
        print("-" * 80)

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
        self.dbg_print_meta()

        print(f"Shape : {self.data.shape}")
        print(f"data : \n{self.data}")
        print(f"groupdq : \n{self.groupdq}")
        # print(f"pixeldq : \n{self.pixeldq}")
        print("-" * 80)

    def dbg_print_pixel_info(self, row, col):
        print("-" * 80)
        print(f"    data")
        for integ in range(self.data.shape[0]):
            print(f"[{integ}] {self.data[integ, :, row, col]}")
        print(f"    groupdq")
        for integ in range(self.data.shape[0]):
            print(f"[{integ}] {self.groupdq[integ, :, row, col]}")
        # print(f"    pixeldq :\n{self.pixeldq[row, col]}")

    def dbg_print_info(self):
        print(" ")
        nints, ngroups, nrows, ncols = self.data.shape
        for row in range(nrows):
            for col in range(ncols):
                print("=" * 80)
                print(f"**** Pixel ({row}, {col}) ****")
                self.dbg_print_pixel_info(row, col)
        print("=" * 80)

    def dbg_write_ramp_data_pix_pre(self, fname, row, col, fd):
        fd.write("def create_ramp_data_pixel():\n")
        indent = INDENT
        fd.write(f"{indent}'''\n")
        fd.write(f"{indent}Using pixel ({row}, {col})\n")
        fd.write(f"{indent}'''\n")
        fd.write(f"{indent}ramp_data = RampData()\n\n")

        fd.write(f"{indent}ramp_data.instrument_name = '{self.instrument_name}'\n\n")

        fd.write(f"{indent}ramp_data.frame_time = {self.frame_time}\n")
        fd.write(f"{indent}ramp_data.group_time = {self.group_time}\n")
        fd.write(f"{indent}ramp_data.groupgap = {self.groupgap}\n")
        fd.write(f"{indent}ramp_data.nframes = {self.nframes}\n")
        fd.write(f"{indent}ramp_data.drop_frames1 = {self.drop_frames1}\n\n")

        fd.write(f"{indent}ramp_data.flags_do_not_use = {self.flags_do_not_use}\n")
        fd.write(f"{indent}ramp_data.flags_jump_det = {self.flags_jump_det}\n")
        fd.write(f"{indent}ramp_data.flags_saturated = {self.flags_saturated}\n")
        fd.write(f"{indent}ramp_data.flags_no_gain_val = {self.flags_no_gain_val}\n")
        fd.write(f"{indent}ramp_data.flags_unreliable_slope = {self.flags_unreliable_slope}\n\n")


        fd.write(f"{indent}ramp_data.start_row = 0\n")
        fd.write(f"{indent}ramp_data.num_rows = 1\n\n")

        fd.write(f"{indent}ramp_data.suppress_one_group_ramps = {self.suppress_one_group_ramps}\n\n")

        nints, ngroups, nrows, ncols = self.data.shape
        fd.write(f"{indent}data = np.zeros(({nints}, {ngroups}, 1, 1), dtype=np.float32)\n")
        fd.write(f"{indent}gdq = np.zeros(({nints}, {ngroups}, 1, 1), dtype=np.uint8)\n")
        fd.write(f"{indent}pdq = np.zeros((1, 1), dtype=np.uint32)\n")


    def dbg_write_ramp_data_pix_post(self, fname, row, col, fd):
        indent = INDENT

        fd.write(f"{indent}ramp_data.data = data\n")
        fd.write(f"{indent}ramp_data.groupdq = gdq\n")
        fd.write(f"{indent}ramp_data.pixeldq = pdq\n")
        fd.write(f"{indent}ramp_data.zeroframe = zframe\n\n")

        fd.write(f"{indent}return ramp_data, ngain, nrnoise\n")

    def dbg_write_ramp_data_pix_pixel(self, fname, row, col, gain, rnoise, fd):
        import numpy as np
        indent = INDENT

        # XXX Make this a separate function
        delimiter = "-" * 40
        fd.write(f"{indent}# {delimiter}\n\n");
        fd.write(f"{indent}# ({row}, {col})\n\n");

        nints = self.data.shape[0]

        for integ in range(nints):
            arr_str = np.array2string(self.data[integ, :, row, col], precision=12, max_line_width=np.nan, separator=", ")
            fd.write(f"{indent}data[{integ}, :, 0, 0] = np.array({arr_str})\n")
        fd.write("\n")

        for integ in range(nints):
            arr_str = np.array2string(self.groupdq[integ, :, row, col], precision=12, max_line_width=np.nan, separator=", ")
            fd.write(f"{indent}gdq[{integ}, :, 0, 0] = np.array({arr_str})\n")
        fd.write("\n")

        arr_str = np.array2string(self.pixeldq[row, col], precision=12, max_line_width=np.nan, separator=", ")
        fd.write(f"{indent}pdq[0, 0] = {arr_str}\n\n")

        if self.zeroframe is not None:
            fd.write(f"{indent}zframe = np.zeros((1, 1), dtype=np.float32)\n\n")
            arr_str = np.array2string(self.zeroframe[row, col], precision=12, max_line_width=np.nan, separator=", ")
            fd.write(f"{indent}zframe[0, 0] = {arr_str}\n\n")
        else:
            fd.write(f"{indent}zframe = None\n\n")

        fd.write(f"{indent}ngain = np.zeros((1, 1), dtype=np.float32)\n")
        fd.write(f"{indent}ngain[0, 0] = {gain[row, col]}\n\n")

        fd.write(f"{indent}nrnoise = np.zeros((1, 1), dtype=np.float32)\n")
        fd.write(f"{indent}nrnoise[0, 0] = {rnoise[row, col]}\n\n")


    def dbg_write_ramp_data_pix(self, fname, row, col, gain, rnoise):
        print(f"*** {fname} ***")
        with open(fname, "w") as fd:
            self.dbg_write_ramp_data_pix_pre(fname, row, col, fd)
            self.dbg_write_ramp_data_pix_pixel(fname, row, col, gain, rnoise, fd)
            self.dbg_write_ramp_data_pix_post(fname, row, col, fd)
