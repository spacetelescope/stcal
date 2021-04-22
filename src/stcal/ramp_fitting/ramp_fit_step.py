#! /usr/bin/env python

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!! Not sure this file should be here !!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from . import ramp_fit

__all__ = ["RampFitStep"]


class RampFitStep:

    """
    This step fits a straight line to the value of counts vs. time to
    determine the mean count rate for each pixel.
    """
    def process(self, input):
        pass
