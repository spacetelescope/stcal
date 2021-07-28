dqflags = {
    "DO_NOT_USE": None,
    "SATURATED": None,
    "JUMP_DET": None,
    "NO_GAIN_VALUE": None,
    "UNRELIABLE_SLOPE": None,
}


def update_dqflags(input_flags):
    dqflags["DO_NOT_USE"] = input_flags["DO_NOT_USE"]
    dqflags["SATURATED"] = input_flags["SATURATED"]
    dqflags["JUMP_DET"] = input_flags["JUMP_DET"]
    dqflags["NO_GAIN_VALUE"] = input_flags["NO_GAIN_VALUE"]
    dqflags["UNRELIABLE_SLOPE"] = input_flags["UNRELIABLE_SLOPE"]


def update_dqflags_from_ramp_data(ramp_data):
    dqflags["DO_NOT_USE"] = ramp_data.flags_do_not_use
    dqflags["SATURATED"] = ramp_data.flags_saturated
    dqflags["JUMP_DET"] = ramp_data.flags_jump_det
    dqflags["NO_GAIN_VALUE"] = ramp_data.flags_no_gain_val
    dqflags["UNRELIABLE_SLOPE"] = ramp_data.flags_unreliable_slope
