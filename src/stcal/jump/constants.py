dqflags = {
    "DO_NOT_USE": None,
    "SATURATED": None,
    "JUMP_DET": None,
    "GOOD": None,
    "NO_GAIN_VALUE": None,
}


def update_dqflags(input_flags):
    dqflags["DO_NOT_USE"] = input_flags["DO_NOT_USE"]
    dqflags["SATURATED"] = input_flags["SATURATED"]
    dqflags["JUMP_DET"] = input_flags["JUMP_DET"]
    dqflags["GOOD"] = input_flags["GOOD"]
    dqflags["NO_GAIN_VALUE"] = input_flags["NO_GAIN_VALUE"]
