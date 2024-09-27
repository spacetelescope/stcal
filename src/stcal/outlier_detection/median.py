"""Compute median of large datasets in memory- and runtime-efficient ways."""

import logging
import tempfile
import warnings
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

_ONE_MB = 1 << 20


def nanmedian3D(cube, overwrite_input=True):
    """Compute the median of a cube ignoring warnings and with
    memory efficiency optimizations. np.nanmedian always uses at least 64-bit
    precision internally, and this is too memory-intensive. Instead, loop over
    the median calculation to avoid the memory usage of the internal upcasting
    and temporary array allocation. The additional runtime of this loop is
    indistinguishable from zero, but this loop cuts overall step memory usage
    roughly in half for at least one test association.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore",
                                message="All-NaN slice encountered",
                                category=RuntimeWarning)
        output_arr = np.empty(cube.shape[1:], dtype=np.float32)
        for i in range(output_arr.shape[0]):
            # this for loop looks silly, but see docstring above
            np.nanmedian(cube[:, i, :],
                         axis=0,
                         overwrite_input=overwrite_input,
                         out=output_arr[i, :])
        return output_arr


def make_median_computer(full_shape, in_memory, buffer_size, dtype):

    if in_memory:
        # allocate memory for data arrays that go into median
        median_computer = np.empty(full_shape, dtype=np.float32)
    else:
        # set up temporary storage for data arrays that go into median
        median_computer = OnDiskMedian(full_shape,
                                       dtype=dtype,
                                       buffer_size=buffer_size)
    return median_computer


def append_to_median_computer(median_computer, idx, data, in_memory):
    if in_memory:
        # populate pre-allocated memory with the drizzled data
        median_computer[idx] = data
    else:
        # distribute the drizzled data into the temporary storage
        median_computer.add_image(data)


def evaluate_median_computer(median_computer, in_memory):
    if in_memory:
        median_data = nanmedian3D(median_computer)
        del median_computer
    else:
        median_data = median_computer.compute_median()
        median_computer.cleanup()
    return median_data


class DiskAppendableArray:
    """
    Creates a temporary file to which to append data, in order to perform
    timewise operations on a stack of input images without holding all of them
    in memory.

    This class is purpose-built for median computation during outlier detection
    and is not very flexible. It is assumed that each data array passed to
    `append` represents the same spatial segment of the full dataset. It is
    also assumed that each array passed to `append` represents only a single
    instant in time; the append operation will stack them along a new axis.

    The `read` operation is only capable of loading the full array back
    into memory. When working with large datasets that do not fit in memory
    the required workflow is to create many DiskAppendableArray objects, each
    holding a small spatial segment of the full dataset.
    """

    def __init__(self, slice_shape, dtype, filename):
        """
        Parameters
        ----------
        slice_shape : tuple
            The shape of the spatial section of input data to be appended
            to the array.

        dtype : str
            The data type of the array. Must be a valid numpy array datatype.

        filename : str
            The full file path in which to store the array
        """
        if len(slice_shape) != 2:
            msg = f"Invalid slice_shape {slice_shape}. Only 2-D arrays "
            msg += "are supported."
            raise ValueError(msg)
        self._filename = Path(filename)
        with Path.open(filename, "wb") as f:   # noqa: F841
            pass
        self._slice_shape = slice_shape
        self._dtype = np.dtype(dtype)
        self._append_count = 0

    @property
    def shape(self):
        return (self._append_count, *self._slice_shape)

    def append(self, data):
        """Add a new slice to the temporary file."""
        if data.shape != self._slice_shape:
            msg = f"Data shape {data.shape} does not match slice shape "
            msg += f"{self._slice_shape}"
            raise ValueError(msg)
        if data.dtype != self._dtype:
            msg = f"Data dtype {data.dtype} does not match array dtype "
            msg += f"{self._dtype}"
            raise ValueError(msg)
        with Path.open(self._filename, "ab") as f:
            data.tofile(f, sep="")
        self._append_count += 1

    def read(self):
        """Read the 3-D array into memory."""
        shp = (self._append_count, *self._slice_shape)
        with Path.open(self._filename, "rb") as f:
            return np.fromfile(f, dtype=self._dtype).reshape(shp)


class OnDiskMedian:

    def __init__(self, shape, dtype="float32", tempdir="", buffer_size=None):
        """
        Set up temporary files to perform operations on a stack of 2-D input
        arrays along the stacking axis (e.g., a time axis) without
        holding all of them in memory. Currently the only supported operation
        is the median.

        Parameters
        ----------
        shape: tuple
            The shape of the entire input, (n_images, imrows, imcols).

        dtype : str
            The data type of the input data.

        tempdir : str
            The parent directory in which to create the temporary directory,
            which itself holds all the DiskAppendableArray tempfiles.
            Default is the current working directory.

        buffer_size : int, optional
            The buffer size, units of bytes.
            Default is the size of one input image.
        """
        if len(shape) != 3:
            msg = f"Invalid input shape {shape}; "
            msg += "only three-dimensional data are supported."
            raise ValueError(msg)
        self._expected_nframes = shape[0]
        self.frame_shape = shape[1:]
        self.dtype = np.dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self._temp_dir = tempfile.TemporaryDirectory(dir=tempdir)
        self._temp_path = Path(self._temp_dir.name)

        # figure out number of sections and rows per section that are needed
        self.nsections, self.section_nrows = \
            self._get_buffer_indices(buffer_size=buffer_size)
        self.slice_shape = (self.section_nrows, shape[2])
        self._n_adds = 0

        # instantiate a temporary DiskAppendableArray for each section
        self._temp_arrays = self._temparray_setup(dtype)

    def _get_buffer_indices(self, buffer_size=None):
        """
        Parameters
        ----------
        buffer_size : int, optional
            The buffer size for the median computation, units of bytes.

        Returns
        -------
        nsections : int
            The number of sections to divide the input data into.

        section_nrows : int
            The number of rows in each section (except the last one).
        """
        imrows, imcols = self.frame_shape
        if buffer_size is None:
            buffer_size = imrows * imcols * self.itemsize
        per_model_buffer_size = buffer_size / self._expected_nframes
        min_buffer_size = imcols * self.itemsize
        section_nrows = \
            min(imrows, int(per_model_buffer_size // min_buffer_size))

        if section_nrows <= 0:
            buffer_size = min_buffer_size * self._expected_nframes
            msg = "Buffer size is too small to hold a single row. "
            msg += f"Increasing buffer size to {buffer_size / _ONE_MB} MB"
            log.warning(msg)
            section_nrows = 1
        self.buffer_size = buffer_size

        nsections = int(np.ceil(imrows / section_nrows))
        msg = f"Computing median over {self._expected_nframes} "
        msg += f"groups in {nsections} sections "
        msg += f"with total memory buffer {buffer_size / _ONE_MB} MB"
        log.info(msg)
        return nsections, section_nrows

    def _temparray_setup(self, dtype):
        """Set up temp file handlers for each spatial section."""
        temp_arrays = []
        for i in range(self.nsections):
            shp = self.slice_shape
            if i == self.nsections - 1:
                # last section has whatever shape is left over
                shp = (self.frame_shape[0] - (self.nsections-1) *
                       self.section_nrows, self.frame_shape[1])
            arr = DiskAppendableArray(shp, dtype, self._temp_path / f"{i}.bin")
            temp_arrays.append(arr)
        return temp_arrays

    def add_image(self, data):
        """
        Split resampled model data into spatial sections
        and write to disk.
        """
        if self._n_adds >= self.nsections:
            msg = "Too many calls to add_image. "
            msg += f"Expected at most {self.nsections} input models."
            raise IndexError(msg)
        self._validate_data(data)
        self._n_adds += 1
        for i in range(self.nsections):
            row1 = i * self.section_nrows
            row2 = min(row1 + self.section_nrows, self.frame_shape[0])
            arr = self._temp_arrays[i]
            arr.append(data[row1:row2])

    def _validate_data(self, data):
        if data.shape != self.frame_shape:
            msg = f"Data shape {data.shape} does not match expected shape "
            msg += f"{self.frame_shape}"
            raise ValueError(msg)
        if data.dtype != self.dtype:
            msg = f"Data dtype {data.dtype} does not match expected dtype "
            msg += f"{self.dtype}"
            raise ValueError(msg)

    def cleanup(self):
        """Remove the temporary files and directory when finished."""
        self._temp_dir.cleanup()

    def compute_median(self):
        """
        Read spatial sections from disk and compute the median across groups
        (median over number of exposures on a per-pixel basis).
        """
        row_indices = [(i * self.section_nrows,
                        min((i+1) * self.section_nrows, self.frame_shape[0]))
                       for i in range(self.nsections)]

        output_rows = row_indices[-1][1]
        output_cols = self._temp_arrays[0].shape[2]
        median_image = np.full((output_rows, output_cols),
                               np.nan,
                               dtype=self.dtype)

        for i, disk_arr in enumerate(self._temp_arrays):
            row1, row2 = row_indices[i]
            arr = disk_arr.read()
            median_image[row1:row2] = nanmedian3D(arr)
            del arr, disk_arr

        return median_image
