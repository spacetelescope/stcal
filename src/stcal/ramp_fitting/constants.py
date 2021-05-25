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
