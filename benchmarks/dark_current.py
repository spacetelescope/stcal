import numpy

from stcal.dark_current.dark_sub import do_correction_data as darkcorr
from tests.test_dark_current import make_rampmodel, make_darkmodel, dqflags


class TimeSuite:
    def setup_cache(self):
        return make_rampmodel(), make_darkmodel()

    def time_more_sci_frames(self, models):
        """
        Check that data is unchanged if there are more frames in the science
        data is than in the dark reference file and verify that when the dark
        is not applied, the data is correctly flagged as such
        """

        rampmodel, darkmodel = models

        # ncols of integration
        nints, ngroups, nrows, ncols = 1, 7, 200, 200

        # create raw input data for step
        dm_ramp = rampmodel(nints, ngroups, nrows, ncols)
        dm_ramp.exp_nframes = 1
        dm_ramp.exp_groupgap = 0

        # populate data array of science cube
        for i in range(0, ngroups - 1):
            dm_ramp.data[0, i] = i

        refgroups = 5
        # create dark reference file model with fewer frames than science data
        dark = darkmodel(refgroups, nrows, ncols)

        # populate data array of reference file
        for i in range(0, refgroups - 1):
            dark.data[0, i] = i * 0.1

        # apply correction
        darkcorr(dm_ramp, dark)

    def time_sub_by_frame(self, models):
        """
        Check that if NFRAMES=1 and GROUPGAP=0 for the science data,
        the dark reference data are directly subtracted frame by frame
        """

        rampmodel, darkmodel = models

        # size of integration
        nints, ngroups, nrows, ncols = 1, 10, 200, 200

        # create raw input data for step
        dm_ramp = rampmodel(nints, ngroups, nrows, ncols)
        dm_ramp.exp_nframes = 1
        dm_ramp.exp_groupgap = 0

        # populate data array of science cube
        for i in range(0, ngroups - 1):
            dm_ramp.data[0, i] = i

        # create dark reference file model with more frames than science data
        refgroups = 15
        dark = darkmodel(refgroups, nrows, ncols)

        # populate data array of reference file
        for i in range(0, refgroups - 1):
            dark.data[0, i] = i * 0.1

        # apply correction
        darkcorr(dm_ramp, dark)

    def time_nan(self, models):
        """
        Verify that when a dark has NaNs, these are correctly
        assumed as zero and the PIXELDQ is set properly
        """

        rampmodel, darkmodel = models

        # size of integration
        nints, ngroups, nrows, ncols = 1, 10, 200, 200

        # create raw input data for step
        dm_ramp = rampmodel(nints, ngroups, nrows, ncols)
        dm_ramp.exp_nframes = 1
        dm_ramp.exp_groupgap = 0

        # populate data array of science cube
        for i in range(0, ngroups - 1):
            dm_ramp.data[0, i, :, :] = i

        # create dark reference file model with more frames than science data
        refgroups = 15
        dark = darkmodel(refgroups, nrows, ncols)

        # populate data array of reference file
        for i in range(0, refgroups - 1):
            dark.data[0, i] = i * 0.1

        # set NaN in dark file

        dark.data[0, 5, 100, 100] = numpy.nan

        # apply correction
        darkcorr(dm_ramp, dark)

    def time_dq_combine(self, models):
        """
        Verify that the DQ array of the dark is correctly
        combined with the PIXELDQ array of the science data.
        """

        rampmodel, darkmodel = models

        # size of integration
        nints, ngroups, nrows, ncols = 1, 5, 200, 200

        # create raw input data for step
        dm_ramp = rampmodel(nints, ngroups, nrows, ncols)
        dm_ramp.exp_nframes = 1
        dm_ramp.exp_groupgap = 0

        # populate data array of science cube
        for i in range(1, ngroups - 1):
            dm_ramp.data[0, i, :, :] = i

        # create dark reference file model with more frames than science data
        refgroups = 7
        dark = darkmodel(refgroups, nrows, ncols)

        # populate dq flags of sci pixeldq and reference dq
        dm_ramp.pixeldq[50, 50] = dqflags["JUMP_DET"]
        dm_ramp.pixeldq[50, 51] = dqflags["SATURATED"]

        dark.groupdq[0, 0, 50, 50] = dqflags["DO_NOT_USE"]
        dark.groupdq[0, 0, 50, 51] = dqflags["DO_NOT_USE"]

        # run correction step
        darkcorr(dm_ramp, dark)

    def time_frame_avg(self, models):
        """
        Check that if NFRAMES>1 or GROUPGAP>0, the frame-averaged dark data are
        subtracted group-by-group from science data groups and the ERR arrays
        are not modified
        """

        rampmodel, darkmodel = models

        # size of integration
        nints, ngroups, nrows, ncols = 1, 5, 1024, 1032

        # create raw input data for step
        dm_ramp = rampmodel(nints, ngroups, nrows, ncols)
        dm_ramp.exp_nframes = 4
        dm_ramp.exp_groupgap = 0

        # populate data array of science cube
        for i in range(0, ngroups - 1):
            dm_ramp.data[:, i] = i + 1

        # create dark reference file model

        refgroups = 20  # This needs to be 20 groups for the calculations to work
        dark = darkmodel(refgroups, nrows, ncols)

        # populate data array of reference file
        for i in range(0, refgroups - 1):
            dark.data[0, i] = i * 0.1

        # apply correction
        darkcorr(dm_ramp, dark)
