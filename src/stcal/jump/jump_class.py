INDENT = "    "
DELIM = "-" * 80

class JumpData:
    """Contains data needed for detecting jumps."""

    def __init__(self, jump_model=None, gain2d=None, rnoise2d=None, dqflags=None):
        """
        Initialize JumpData instance.

        Parameters
        ----------
        jump_model : model
            Input data model, assumed to be of type RampModel.

        gain2d : numpy
            The gain for all pixels.

        rnoise2d : numpy
            The read noise for all pixels.

        dqflags : dict
            A dictionary with at least the following keywords:
            DO_NOT_USE, SATURATED, JUMP_DET, NO_GAIN_VALUE, GOOD
        """
        # Get model information

        # The number of frames per group
        self.nframes = 0

        # These are the input arrays
        # data is a 4-D ndarray of the science data
        # gdq is a 4-D ndarray of the group DQ
        # gdq is a 2-D ndarray of the pixel DQ
        self.data, self.gdq, self.pdq = None, None, None  # indata
        if jump_model is not None:
            self.init_arrays_from_model(jump_model)
            self.nframes = jump_model.meta.exposure.nframes

        # Get reference arrays
        # gain is a 2-D ndarray for the gain for each pixel
        self.gain_2d = None
        if gain2d is not None:
            self.gain_2d = gain2d

        # rnoise_2d is a 2-D ndarray for the read noise for each pixel
        self.rnoise_2d = None
        if rnoise2d is not None:
            self.rnoise_2d = rnoise2d

        # Detection options (using JWST defaults)
        # The 'normal' cosmic ray sigma rejection threshold for ramps with more than 4 groups
        self.rejection_thresh = 4.

        # Cosmic ray sigma rejection threshold for ramps having 3 groups
        self.three_grp_thresh = 6.

        # Cosmic ray sigma rejection threshold for ramps having 4 groups
        self.four_grp_thresh = 5.

        # If set to True (default is True), it will cause the four perpendicular
        # neighbors of all detected jumps to also be flagged as a jump.
        self.flag_4_neighbors = True

        # Value in units of sigma that sets the upper limit for flagging of neighbors.
        # Any jump above this cutoff will not have its neighbors flagged.
        self.max_jump_to_flag_neighbors = 1000

        # Value in units of sigma that sets the lower limit for flagging of neighbors (marginal
        # detections). Any primary jump below this value will not have its neighbors flagged.
        self.min_jump_to_flag_neighbors = 10

        # The group DQ flags.
        self.fl_good, self.fl_sat, self.jump = None, None, None
        self.fl_ngv, self.fl_dnu, self.ref = None, None, None
        if dqflags is not None:
            self.fl_good = dqflags["GOOD"]
            self.fl_sat = dqflags["SATURATED"]
            self.fl_jump = dqflags["JUMP_DET"]
            self.fl_ngv = dqflags["NO_GAIN_VALUE"]
            self.fl_dnu = dqflags["DO_NOT_USE"]
            self.fl_ref = dqflags["REFERENCE_PIXEL"]

        # Set default values (JWST defaults)

        # After jump flagging
        # Jumps with amplitudes above the specified DN value will have subsequent
        # groups flagged with the number determined by the after_jump_flag_n1
        self.after_jump_flag_dn1 = 0.0

        # Gives the number of groups to flag after jumps with DN values above that
        # given by after_jump_flag_dn1
        self.after_jump_flag_n1 = 0

        # Jumps with amplitudes above the specified DN value will have subsequent
        # groups flagged with the number determined by the after_jump_flag_n2
        self.after_jump_flag_dn2 = 0.0

        # ives the number of groups to flag after jumps with DN values above that
        # given by after_jump_flag_dn2
        self.after_jump_flag_n2 = 0

        # Computed later, depends on the after flagging above.
        self.after_jump_flag_e1 = None
        self.after_jump_flag_e2 = None

        # Snowball information for near-IR
        #  Turns on Snowball detector for NIR detectors
        self.expand_large_events = False

        # The minimum contour area to trigger the creation of enclosing ellipses or circles.
        self.min_jump_area = (5,)

        # The minimum area of saturated pixels at the center of a snowball. Only
        # contours with area above the minimum will create snowballs.
        self.min_sat_area = 1

        # The factor that is used to increase the size of the enclosing
        # circle/ellipse jump flagged pixels.
        self.expand_factor = 2.0

        # Deprecated
        self.use_ellipses = False

        # If true there must be a saturation circle within the radius of the jump
        # circle to trigger the creation of a snowball. All true snowballs appear
        # to have at least one saturated pixel.
        self.sat_required_snowball = True

        # The minimum radius of the saturated core of a snowball for the core to be extended
        self.min_sat_radius_extend = 2.5

        # The number of pixels to expand the saturated core of detected snowballs
        self.sat_expand = 2

        # The distance from the edge of the detector where saturated cores are not
        # required for snowball detection
        self.edge_size = 25

        # MIRI shower information

        # Turns on the flagging of the faint extended emission of MIRI showers
        self.find_showers = False

        # The SNR minimum for the detection of faint extended showers in MIRI
        self.extend_snr_threshold = 1.2

        # The required minimum area of extended emission after convolution for the
        # detection of showers in MIRI
        self.extend_min_area = 90

        # The inner radius of the Ring2DKernal that is used for the detection of
        # extended emission in showers
        self.extend_inner_radius = 1

        # The outer radius of the Ring2DKernal that is used for the detection of
        # extended emission in showers
        self.extend_outer_radius = 2.6

        # Multiplicative factor to expand the radius of the ellipse fit to the
        # detected extended emission in MIRI showers
        self.extend_ellipse_expand_ratio = 1.2

        # The minimum number of groups to switch to flagging all outliers in a single pass.
        self.min_diffs_single_pass = 10

        # The maximum possible amplitude for flagged MIRI showers in DN/group
        self.max_shower_amplitude = 6

        # The maximum radius for any extension of saturation or jump
        self.max_extended_radius = 200

        # Sigma clipping
        # The minimum number of groups for jump detection
        self.minimum_groups = 3

        # The minimum number of groups required to use sigma clipping to find outliers.
        self.minimum_sigclip_groups = 100

        # In sigma clipping, if True only differences between integrations are compared.
        # If False, then all differences are processed at once.
        self.only_use_ints = True

        # Internal state
        # Number of groups after detected extended emission to flag as a jump for MIRI showers
        self.grps_masked_after_shower = 5

        # The flag to turn on the extension of the flagging of the saturated cores of snowballs.
        self.mask_persist_grps_next_int = True

        # How many groups to be flagged when the saturated cores are extended into
        # subsequent integrations.
        self.persist_grps_flagged = 25

        # Multiprocessing, data a sliced by row
        self.max_cores = None  # Number of processes
        self.start_row = 0  # Start row of current slice
        self.end_row = 0  # End row of current slice
        self.tot_row = 0 # Total number of rows in slice

    def init_arrays_from_model(self, jump_model):
        """
        Set arrays from a data model.

        Parameters
        ----------
        jump_model : datamodel
            A datamodel with certain expected parameters.
        """
        self.data = jump_model.data
        self.gdq = jump_model.groupdq
        self.pdq = jump_model.pixeldq

    def init_arrays_from_arrays(self, data, gdq, pdq):
        """
        Set arrays from a numpy arrays.

        Parameters
        ----------
        data : ndarray
            The 4-D science array

        gdq : ndarray
            The 4-D group DQ array

        pdq : ndarray
            The 2-D pixel DQ array
        """
        self.data = data
        self.gdq = gdq
        self.pdq = pdq

    def set_detection_settings(self, rej, _3grp, _4grp, mx, mn, _4flag):
        """
        Set class instance detection settings.

        Parameters
        ----------
        rej : float
            The 'normal' cosmic ray sigma rejection threshold for ramps with more
            than 4 groups

        _3grp : float
            cosmic ray sigma rejection threshold for ramps having 3 groups

        _4grp : float
            cosmic ray sigma rejection threshold for ramps having 4 groups

        mx : float
            value in units of sigma that sets the upper limit for flagging of
            neighbors. Any jump above this cutoff will not have its neighbors
            flagged.

        mn : float
            value in units of sigma that sets the lower limit for flagging of
            neighbors (marginal detections). Any primary jump below this value will
            not have its neighbors flagged.

        _4flag : bool
            if set to True (default is True), it will cause the four perpendicular
            neighbors of all detected jumps to also be flagged as a jump.
        """
        self.rejection_thresh = rej
        self.three_grp_thresh = _3grp
        self.four_grp_thresh = _4grp
        self.max_jump_to_flag_neighbors = mx
        self.min_jump_to_flag_neighbors = mn
        self.flag_4_neighbors = _4flag

    def set_after_jump(self, dn1, n1, dn2, n2):
        """
        Set class instance after jump values.

        Parameters
        ----------
        dn1 : float
            Jumps with amplitudes above the specified DN value will have subsequent
            groups flagged with the number determined by the after_jump_flag_n1

        n1 : int
            Gives the number of groups to flag after jumps with DN values above that
            given by after_jump_flag_dn1

        dn2 : float
            Jumps with amplitudes above the specified DN value will have subsequent
            groups flagged with the number determined by the after_jump_flag_n2

        n2 : int
            Gives the number of groups to flag after jumps with DN values above that
            given by after_jump_flag_dn2
        """
        self.after_jump_flag_dn1 = dn1
        self.after_jump_flag_n1 = n1
        self.after_jump_flag_dn2 = dn2
        self.after_jump_flag_n2 = n2

    def set_snowball_info(
            self, levent, mjarea, msarea, exfact, require, satrad, satexp, edge):
        """
        Set class instance values needed for snowball handling.

        Parameters
        ----------
        levent : bool
            When True this triggers the flagging of snowballs for NIR detectors.

        mjarea : float
            The minimum contour area to trigger the creation of enclosing ellipses
            or circles.

        msarea : int
            The minimum area of saturated pixels within the jump circle to trigger
            the creation of a snowball.

        expfact : float
            The factor that increases the size of the snowball or enclosing ellipse.

        require : bool
            If true there must be a saturation circle within the radius of the jump
            circle to trigger the creation of a snowball. All true snowballs appear
            to have at least one saturated pixel.

        satrad : float
            The minimum radius of the saturated core of a snowball for the core to
            be extended

        satexp : float
            The number of pixels to expand the saturated core of detected snowballs

        edge : int
            The distance from the edge of the detector where saturated cores are not required for snowball detection
        """
        self.expand_large_events = levent
        self.min_jump_area = mjarea
        self.min_sat_area = msarea
        self.expand_factor = exfact
        self.sat_required_snowball = require
        self.min_sat_radius_extend = satrad
        self.sat_expand = satexp
        self.edge_size = edge

    def set_shower_info(self, shower, snr, marea, inner, outer, expand, single, extend):
        """
        Set class instance values needed for shower handling.

        Parameters
        ----------
        showers : boolean
            Turns on the flagging of the faint extended emission of MIRI showers

        snr : float
            The SNR minimum for the detection of faint extended showers in MIRI

        marea : float
            The required minimum area of extended emission after convolution for the
            detection of showers in MIRI

        inner : float
            The inner radius of the Ring2DKernal that is used for the detection of
            extended emission in showers

        outer : float
            The outer radius of the Ring2DKernal that is used for the detection of
            extended emission in showers

        expand : float
            Multiplicative factor to expand the radius of the ellipse fit to the
            detected extended emission in MIRI showers

        single : int
           The minimum number of groups to switch to flagging all outliers in a single pass.

        extend : int
            The maximum radius for any extension of saturation or jump
        """
        # MIRI shower information
        self.find_showers = shower
        self.extend_snr_threshold = snr
        self.extend_min_area = marea
        self.extend_inner_radius = inner
        self.extend_outer_radius = outer
        self.extend_ellipse_expand_ratio = expand
        self.min_diffs_single_pass = single
        self.max_extended_radius = extend

    def set_sigma_clipping_info(self, mingrps, minsig, useints):
        """
        Set class instance values needed for sigma clipping.

        Parameters
        ----------
        mingrps : int
           The minimum number of groups for jump detection

        minsig : int
            The minimum number of groups required to use sigma clipping to find outliers.

        useints : boolean
            In sigma clipping, if True only differences between integrations are compared.
            If False, then all differences are processed at once.
        """
        # Sigma clipping
        self.minimum_groups = mingrps
        self.minimum_sigclip_groups = minsig
        self.only_use_ints = useints  # XXX

    # --------------------------------------------------------------------------
    # Print methods, primarily used for debugging.

    def print_jump_data(self, fd=None):
        """Print jump_data information."""
        self.print_jump_data_arrays(fd=fd)
        self.print_jump_data_options(fd=fd)
        self.print_jump_data_dqflags(fd=fd)
        self.print_jump_data_snowball(fd=fd)
        self.print_jump_data_shower(fd=fd)
        self.print_jump_data_sigma_clipping(fd=fd)
        self.print_jump_data_internal_state(fd=fd)

    def print_jump_data_arrays(self, pix_list=None, fd=None):
        """Print jump_data arrays."""
        if fd is None:
            print(self.get_jump_data_arrays())
        else:
            print(self.get_jump_data_arrays(), file=fd)


    def get_jump_data_arrays(self):
        """Return string of jump_data information."""
        oline = f"{DELIM}\n"
        oline += "JumpData Arrays\n"
        oline += f"Data shape = {self.data.shape}\n"
        '''
        self.data
        self.gdq
        self.pdq
        self.gain_2d
        self.rnoise_2d
        '''
        oline += f"{DELIM}\n"
        return oline

    def print_jump_data_options(self, fd=None):
        """Print jump_data options."""
        if fd is None:
            print(self.get_jump_data_options())
        else:
            print(self.get_jump_data_options(), file=fd)

    def get_jump_data_options(self):
        """Return string of jump_data option."""
        oline = f"{DELIM}\n"
        oline += f"JumpData Options\n"
        oline += f"nframes = {self.nframes}\n"
        oline += f"rejection_thresh = {self.rejection_thresh}\n"
        oline += f"three_grp_thresh = {self.three_grp_thresh}\n"
        oline += f"four_grp_thresh = {self.four_grp_thresh}\n"
        oline += f"flag_4_neighbors = {self.flag_4_neighbors}\n"
        oline += f"max_jump_to_flag_neighbors = {self.max_jump_to_flag_neighbors}\n"
        oline += f"min_jump_to_flag_neighbors = {self.min_jump_to_flag_neighbors}\n\n"

        # After jump flagging
        oline += "After Jump Flags\n"
        oline += f"{INDENT}after_jump_flag_dn1 = {self.after_jump_flag_dn1}\n"
        oline += f"{INDENT}after_jump_flag_n1 = {self.after_jump_flag_n1}\n"
        oline += f"{INDENT}after_jump_flag_dn2 = {self.after_jump_flag_dn2}\n"
        oline += f"{INDENT}after_jump_flag_n2 = {self.after_jump_flag_n2}\n\n"

        # Computed later, depends on the after flagging above.
        oline += f"{INDENT}after_jump_flag_e1 = {self.after_jump_flag_e1}\n"
        oline += f"{INDENT}after_jump_flag_e2 = {self.after_jump_flag_e2}\n"
        oline += f"{DELIM}\n"
        return oline

    def print_jump_data_dqflags(self, fd=None):
        """Print string of jump_data dqflags."""
        if fd is None:
            print(self.get_jump_data_dqflags())
        else:
            print(self.get_jump_data_dqflags(), file=fd)

    def get_jump_data_dqflags(self):
        """Return string of jump_data dqflags."""
        oline = f"{DELIM}\n"
        oline += "DQ Flags\n"
        oline += f"{INDENT}fl_good = {self.fl_good}\n"
        oline += f"{INDENT}fl_sat = {self.fl_sat}\n"
        oline += f"{INDENT}fl_jump = {self.fl_jump}\n"
        oline += f"{INDENT}fl_ngv = {self.fl_ngv}\n"
        oline += f"{INDENT}fl_dnu = {self.fl_dnu}\n"
        oline += f"{INDENT}fl_ref = {self.fl_ref}\n"
        oline += f"{DELIM}\n\n"
        return oline

    def print_jump_data_snowball(self, fd=None):
        """Print string of jump_data snowball data."""
        if fd is None:
            print(self.get_jump_data_snowball())
        else:
            print(self.get_jump_data_snowball(), file=fd)

    def get_jump_data_snowball(self):
        """Return string of jump_data snowball data."""
        oline = f"{DELIM}\n"
        oline += "Snowball Information\n"
        oline += f"expand_large_events = {self.expand_large_events}\n"
        oline += f"min_jump_area = {self.min_jump_area}\n"
        oline += f"min_sat_area = {self.min_sat_area}\n"
        oline += f"expand_factor = {self.expand_factor}\n"
        oline += f"use_ellipses (deprecated) = {self.use_ellipses}\n"
        oline += f"sat_required_snowball = {self.sat_required_snowball}\n"
        oline += f"min_sat_radius_extend = {self.min_sat_radius_extend}\n"
        oline += f"sat_expand = {self.sat_expand}\n"
        oline += f"edge_size = {self.edge_size}\n"
        oline += f"{DELIM}\n\n"
        return oline

    def print_jump_data_shower(self, fd=None):
        """Print string of jump_data shower data."""
        if fd is None:
            print(self.get_jump_data_shower())
        else:
            print(self.get_jump_data_shower(), file=fd)

    def get_jump_data_shower(self):
        """Return string of jump_data shower data."""
        oline = f"{DELIM}\n"
        oline += "Shower Information\n"
        oline += f"find_showers = {self.find_showers}\n"
        oline += f"extend_snr_threshold = {self.extend_snr_threshold}\n"
        oline += f"extend_min_area = {self.extend_min_area}\n"
        oline += f"extend_inner_radius = {self.extend_inner_radius}\n"
        oline += f"extend_outer_radius = {self.extend_outer_radius}\n"
        oline += f"extend_ellipse_expand_ratio = {self.extend_ellipse_expand_ratio}\n"
        oline += f"min_diffs_single_pass = {self.min_diffs_single_pass}\n"
        oline += f"max_extended_radius = {self.max_extended_radius}\n"
        oline += f"{DELIM}\n\n"
        return oline

    def print_jump_data_sigma_clipping(self, fd=None):
        """Print string of jump_data sigma clipping data."""
        if fd is None:
            print(self.get_jump_data_sigma_clipping())
        else:
            print(self.get_jump_data_sigma_clipping(), file=fd)

    def get_jump_data_sigma_clipping(self):
        """Return string of jump_data sigma clipping data."""
        oline = f"{DELIM}\n"
        oline += "Sigma Clipping\n"
        oline += f"minimum_groups = {self.minimum_groups}\n"
        oline += f"minimum_sigclip_groups = {self.minimum_sigclip_groups}\n"
        oline += f"only_use_ints = {self.only_use_ints}\n"
        oline += f"{DELIM}\n\n"
        return oline

    def print_jump_data_internal_state(self, fd=None):
        """Print string of jump_data internal state."""
        if fd is None:
            print(self.get_jump_data_internal_state())
        else:
            print(self.get_jump_data_internal_state(), file=fd)

    def get_jump_data_internal_state(self):
        """Return string of jump_data internal state."""
        oline = f"{DELIM}\n"
        oline += "Internal State\n"
        oline += f"grps_masked_after_shower = {self.grps_masked_after_shower}\n"
        oline += f"mask_persist_grps_next_int = {self.mask_persist_grps_next_int}\n"
        oline += f"persist_grps_flagged = {self.persist_grps_flagged}\n"
        oline += f"{DELIM}\n\n"
        return oline
