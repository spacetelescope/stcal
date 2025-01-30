class TwoPointParams:
    """Contains data needed for computing two point difference."""

    def __init__(self, jump_data=None, copy_arrs=False):
        """
        Initialize TwoPointParams instance.

        Parameters
        ----------
        jump_data : JumpData
            Class containing parameters and methods to detect jumps.

        copy_arrs : bool
            Should the arrays be copied.
        """
        if jump_data is not None:
            self.normal_rej_thresh = jump_data.rejection_thresh

            self.two_diff_rej_thresh = jump_data.three_grp_thresh
            self.three_diff_rej_thresh = jump_data.four_grp_thresh
            self.nframes = jump_data.nframes,
                
            self.flag_4_neighbors = jump_data.flag_4_neighbors
            self.max_jump_to_flag_neighbors = jump_data.max_jump_to_flag_neighbors
            self.min_jump_to_flag_neighbors = jump_data.min_jump_to_flag_neighbors

            self.fl_good = jump_data.fl_good
            self.fl_sat = jump_data.fl_sat
            self.fl_jump = jump_data.fl_jump
            self.fl_ngv = jump_data.fl_ngv
            self.fl_dnu = jump_data.fl_dnu
            self.fl_ref = jump_data.fl_ref

            self.after_jump_flag_e1=jump_data.after_jump_flag_e1
            self.after_jump_flag_n1=jump_data.after_jump_flag_n1
            self.after_jump_flag_e2=jump_data.after_jump_flag_e2
            self.after_jump_flag_n2=jump_data.after_jump_flag_n2

            self.minimum_groups = jump_data.minimum_groups
            self.minimum_sigclip_groups = jump_data.minimum_sigclip_groups
            self.only_use_ints = jump_data.only_use_ints
            self.min_diffs_single_pass = jump_data.min_diffs_single_pass
        else:
            self.normal_rej_thresh = None

            self.two_diff_rej_thresh = None
            self.three_diff_rej_thresh = None
            self.nframes = None
                
            self.flag_4_neighbors = None
            self.max_jump_to_flag_neighbors = None
            self.min_jump_to_flag_neighbors = None

            self.fl_good = None
            self.fl_sat = None
            self.fl_jump = None
            self.fl_ngv = None
            self.fl_dnu = None
            self.fl_ref = None

            self.after_jump_flag_e1 = None
            self.after_jump_flag_n1 = None
            self.after_jump_flag_e2 = None
            self.after_jump_flag_n2 = None

            self.minimum_groups = None
            self.minimum_sigclip_groups = None
            self.only_use_ints = None
            self.min_diffs_single_pass = None

        self.copy_arrs = copy_arrs

    def __repr__(self):
        """Create __repr__ string."""
        delim = "-" * 60
        ostr = f"{delim}\n"
        ostr += f"normal_rej_thresh = {self.normal_rej_thresh}\n"
        ostr += f"two_diff_rej_thresh = {self.two_diff_rej_thresh}\n"
        ostr += f"three_diff_rej_thresh = {self.three_diff_rej_thresh}\n"
        ostr += f"nframes = {self.nframes}\n\n"
            
        ostr += f"flag_4_neighbors = {self.flag_4_neighbors}\n"
        ostr += f"max_jump_to_flag_neighbors = {self.max_jump_to_flag_neighbors}\n"
        ostr += f"min_jump_to_flag_neighbors = {self.min_jump_to_flag_neighbors}\n\n"

        ostr += f"fl_good = {self.fl_good}\n"
        ostr += f"fl_sat = {self.fl_sat}\n"
        ostr += f"fl_jump = {self.fl_jump}\n"
        ostr += f"fl_ngv = {self.fl_ngv}\n"
        ostr += f"fl_dnu = {self.fl_dnu}\n"
        ostr += f"fl_ref = {self.fl_ref}\n\n"


        ostr += f"after_jump_flag_e1 = {self.after_jump_flag_e1}\n"
        ostr += f"after_jump_flag_n1 = {self.after_jump_flag_n1}\n"
        ostr += f"after_jump_flag_e2 = {self.after_jump_flag_e2}\n"
        ostr += f"after_jump_flag_n2 = {self.after_jump_flag_n2}\n\n"

        ostr += f"minimum_groups = {self.minimum_groups}\n"
        ostr += f"minimum_sigclip_groups = {self.minimum_sigclip_groups}\n"
        ostr += f"only_use_ints = {self.only_use_ints}\n"
        ostr += f"min_diffs_single_pass = {self.min_diffs_single_pass}\n\n"

        ostr += f"copy_arrs = {self.copy_arrs}\n"
        ostr += f"{delim}\n"

        return ostr
