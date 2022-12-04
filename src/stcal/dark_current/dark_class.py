import numpy as np
from astropy import units as u


class DarkData:
    """
    This class contains all data needed to perform the dark current subtraction
    step.
    """
    def __init__(self, dims=None, dark_model=None):
        """
        Creates a class to remove data model dependencies in the internals of
        the dark current code.  The data contained in this class comes from the
        dark reference data file.  If a dark data model is passed as an
        argument the DarkData uses it to create class arrays and set metadata.
        If a dark data model is None, then the DarkData arrays can still be
        created by using the 'dims' argument.  In that situation, no metadata
        will be contained in the class.  If both arguments are None, all arrays
        and metadata are set to None, requiring the instantiator of the class
        to set wanted values.

        Parameters
        ---------
        dims : tuple, optional
            A tuple of integers to describe the dimensions of the arrays used
            during the dark current step.  This argument is only used if the
            'dark_model' argument is None.  If a dark model is not available
            from which to create a DarkData class, but the dimensions of the
            data array are known, then 'dims' is used (the arrays data, groupdq,
            and err are assumed to have the same dimension).


        dark_model : data model, optional
            Input data model, assumed to be a JWST DarkModel like model.  If
            this argument is not None, the DarkData class will have arrasy and
            meta data set based on the arrays in the dark_model.
        """
        if dark_model is not None:
            self.data = dark_model.data
            self.groupdq = dark_model.dq
            self.err = dark_model.err

            self.exp_nframes = dark_model.meta.exposure.nframes
            self.exp_ngroups = dark_model.meta.exposure.ngroups
            self.exp_groupgap = dark_model.meta.exposure.groupgap

        elif dims is not None:
            self.data = np.zeros(dims, dtype=np.float32)
            self.groupdq = np.zeros(dims, dtype=np.uint32)
            self.err = np.zeros(dims, dtype=np.float32)

            self.exp_nframes = None
            self.exp_ngroups = None
            self.exp_groupgap = None

        else:
            self.data = None
            self.groupdq = None
            self.err = None

            self.exp_nframes = None
            self.exp_ngroups = None
            self.exp_groupgap = None

        self.save = False
        self.output_name = None


class ScienceData:
    def __init__(self, science_model=None):
        """
        A class containing all science data needed to subtract the dark current
        from the data.

        Parameters
        ---------
        science_model : data model, optional
            Input data model, assumed to be a JWST RampModel like model.  If
            this is None, then the class instantiator is responsible for
            populating the data.
        """
        if science_model is not None:
            if isinstance(science_model.data,u.Quantity):
                self.data = science_model.data.value
            else:
                self.data = science_model.data
            self.groupdq = science_model.groupdq
            self.pixeldq = science_model.pixeldq

            if isinstance(science_model.err,u.Quantity):
                self.err = science_model.err.value
            else:
                self.err = science_model.err

            self.exp_nframes = science_model.meta.exposure.nframes
            self.exp_groupgap = science_model.meta.exposure.groupgap

            self.cal_step = None
        else:
            self.data = None
            self.groupdq = None
            self.pixeldq = None
            self.err = None

            self.exp_nframes = None
            self.exp_groupgap = None

            self.cal_step = None
