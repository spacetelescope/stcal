"""Compute median of large datasets in memory- and runtime-efficient ways."""

from __future__ import annotations

import logging
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

_ONE_MB = 1 << 20


__all__ = ["MedianComputer", "nanmedian3D"]


def nanmedian3D(cube: np.ndarray, overwrite_input: bool = True) -> np.ndarray:
    """Compute the nanmedian of a cube.

    This produces identical results to np.nanmedian but is much more
    memory efficient for large cubes.

    RuntimeWarnings reporting "All-Nan slice" will be ignored.

    Parameters
    ----------
    cube
        3-dimensional array. Will be modified in-place if `overwrite_input` is True
    overwrite_input
        Passed to ``np.nanmedian``, if True the input cube will be modified

    Returns
    -------
    np.ndarray
        2-dimensional computed median array
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore",
                                message="All-NaN slice encountered",
                                category=RuntimeWarning)
        output_arr = np.empty(cube.shape[1:], dtype=cube.dtype)
        for i in range(output_arr.shape[0]):
            # np.nanmedian allocates lots of memory; this for loop gets around that
            np.nanmedian(cube[:, i, :],
                         axis=0,
                         overwrite_input=overwrite_input,
                         out=output_arr[i, :])
        return output_arr


class MedianComputer:
    """
    Top-level class to treat median computation uniformly, whether in
    memory or on disk.
    """

    def __init__(self,
                 full_shape: tuple,
                 in_memory: bool,
                 buffer_size: int | None = None,
                 dtype: str | np.dtype = "float32",
                 tempdir: str = "",
                 ) -> None:
        """
        Parameters
        ----------
        full_shape
            The shape of the full input dataset.

        in_memory
            Whether to perform the median computation in memory or using
            temporary files on disk to save memory.

        buffer_size
            The buffer size for the median computation, units of bytes.
            Has no effect if `in_memory` is True.

        dtype
            The data type of the input data.

        tempdir
            The parent directory in which to create the temporary directory.
            Default is the current working directory.
        """
        self.full_shape = full_shape
        self.in_memory = in_memory
        if buffer_size is None:
            buffer_size = 0
        self.buffer_size = buffer_size
        self.dtype = dtype
        if self.in_memory:
            computer: Any = np.empty(full_shape, dtype=dtype)
        else:
            computer = _OnDiskMedian(full_shape,
                                    dtype=dtype,
                                    buffer_size=buffer_size,
                                    tempdir=tempdir)
        self._median_computer: Any = computer

    def append(self,
               data: np.ndarray,
               idx: int | None = None
               ) -> None:
        """
        Append data to the median computer.

        Parameters
        ----------
        data
            The data to append to the median computer. Must have shape full_shape[1:].

        idx
            The index at which to append the data. Must be between 0 and full_shape[0].
            Required if using in-memory median computation.
        """
        if self.in_memory:
            if idx is None:
                msg = "Index must be provided when using in-memory median"
                raise ValueError(msg)
            # populate pre-allocated memory with the drizzled data
            self._median_computer[idx] = data
        else:
            # distribute the drizzled data into the temporary storage
            self._median_computer.add_image(data)

    def evaluate(self) -> np.ndarray:
        """
        Compute the median data from the input data.

        Returns
        -------
        np.ndarray
            The median data computed from the input data.
        """
        if self.in_memory:
            median_data = nanmedian3D(self._median_computer)
        else:
            median_data = self._median_computer.compute_median()
            self._median_computer.cleanup()
        return median_data


class _DiskAppendableArray:
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

    def __init__(self,
                 slice_shape: tuple,
                 dtype: str | np.dtype,
                 filename: str | Path
                 ) -> None:
        """
        Parameters
        ----------
        slice_shape
            The required shape of each appended slice.

        dtype
            The data type of the array. Must be a valid numpy array datatype.

        filename
            The full file path in which to store the array
        """
        if len(slice_shape) != 2:
            msg = f"Invalid slice shape {slice_shape}. Only 2-D arrays are supported."
            raise ValueError(msg)
        self._filename = Path(filename)
        with Path.open(self._filename, "wb") as f:   # noqa: F841
            pass
        self._slice_shape = slice_shape
        self._dtype = np.dtype(dtype)
        self._append_count = 0

    @property
    def shape(self) -> tuple:
        return (self._append_count, *self._slice_shape)

    def append(self, data: np.ndarray) -> None:
        """
        Add a new slice to the temporary file.

        Parameters
        ----------
        data
            The data to append to the array. Must have shape ``slice_shape``
            and dtype matching the one provided to ``__init__``.
        """
        if data.shape != self._slice_shape:
            msg = f"Data shape {data.shape} does not match slice shape {self._slice_shape}"
            raise ValueError(msg)
        if data.dtype != self._dtype:
            msg = f"Data dtype {data.dtype} does not match array dtype {self._dtype}"
            raise ValueError(msg)
        with Path.open(self._filename, "ab") as f:
            data.tofile(f, sep="")
        self._append_count += 1

    def read(self) -> np.ndarray:
        """
        Read the 3-D array into memory.

        Returns
        -------
        array
            The full array stored in the temporary file, shape (append_count, *slice_shape).
        """
        shp = (self._append_count, *self._slice_shape)
        with Path.open(self._filename, "rb") as f:
            return np.fromfile(f, dtype=self._dtype).reshape(shp)


class _OnDiskMedian:

    def __init__(self,
                 shape: tuple,
                 dtype: str | np.dtype = "float32",
                 tempdir: str = "",
                 buffer_size: int = 0
                 ) -> None:
        """
        Set up temporary files to perform operations on a stack of 2-D input
        arrays along the first dimension (e.g., a time axis) without
        holding all of them in memory. Currently the only supported operation
        is the median.

        Parameters
        ----------
        shape
            The shape of the entire input, (n_images, imrows, imcols).

        dtype
            The data type of the input data.

        tempdir
            The parent directory in which to create the temporary directory,
            which itself holds all the DiskAppendableArray tempfiles.
            Default is the current working directory.

        buffer_size
            The buffer size, units of bytes.
            Default is the size of one input image.
        """
        if len(shape) != 3:
            msg = f"Invalid input shape {shape}; only three-dimensional data are supported."
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

    def _get_buffer_indices(self,
                            buffer_size: int = 0
                            ) -> tuple[int, int]:
        """
        Determine the number of sections and rows per section needed to
        divide the input data into sections that fit within the specified
        buffer size.

        Parameters
        ----------
        buffer_size
            The buffer size for the median computation, units of bytes.

        Returns
        -------
        nsections
            The number of sections to divide the input data into.

        section_nrows
            The number of rows in each section (except the last one).
        """
        imrows, imcols = self.frame_shape
        if buffer_size == 0:
            buffer_size = imrows * imcols * self.itemsize
        per_model_buffer_size = buffer_size / self._expected_nframes
        min_buffer_size = imcols * self.itemsize
        section_nrows = \
            min(imrows, int(per_model_buffer_size // min_buffer_size))

        if section_nrows <= 0:
            buffer_size = min_buffer_size * self._expected_nframes
            msg = ("Buffer size is too small to hold a single row. "
                   f"Increasing buffer size to {buffer_size / _ONE_MB} MB")
            log.warning(msg)
            section_nrows = 1
        self.buffer_size = buffer_size

        nsections = int(np.ceil(imrows / section_nrows))
        msg = (f"Computing median over {self._expected_nframes} "
               f"groups in {nsections} sections "
               f"with total memory buffer {buffer_size / _ONE_MB} MB")
        log.info(msg)
        return nsections, section_nrows

    def _temparray_setup(self,
                         dtype: str | np.dtype
                         ) -> list[_DiskAppendableArray]:
        """Set up temp file handlers, one for each section."""
        temp_arrays = []
        for i in range(self.nsections):
            shp = self.slice_shape
            if i == self.nsections - 1:
                # last section has whatever shape is left over
                shp = (self.frame_shape[0] - (self.nsections-1) *
                       self.section_nrows, self.frame_shape[1])
            arr = _DiskAppendableArray(shp, dtype, self._temp_path / f"{i}.bin")
            temp_arrays.append(arr)
        return temp_arrays

    def add_image(self, data: np.ndarray) -> None:
        """
        Split data into ``nsections`` sections and write/append each section to
        the corresponding temporary file.
        """
        if self._n_adds >= self.nsections:
            msg = f"Too many calls to add_image. Expected at most {self.nsections} input models."
            raise IndexError(msg)
        self._validate_data(data)
        self._n_adds += 1
        for i in range(self.nsections):
            row1 = i * self.section_nrows
            row2 = min(row1 + self.section_nrows, self.frame_shape[0])
            arr = self._temp_arrays[i]
            arr.append(data[row1:row2])

    def _validate_data(self, data: np.ndarray) -> None:
        """Ensure data array being appended has correct shape and dtype."""
        if data.shape != self.frame_shape:
            msg = f"Data shape {data.shape} does not match expected shape {self.frame_shape}."
            raise ValueError(msg)
        if data.dtype != self.dtype:
            msg = f"Data dtype {data.dtype} does not match expected dtype {self.dtype}."
            raise ValueError(msg)

    def cleanup(self) -> None:
        """
        Remove the temporary files and directory when finished.

        This method should be called when this instance is no longer needed,
        and the instance should not be used after this method is called.
        """
        self._temp_dir.cleanup()

    def compute_median(self) -> np.ndarray:
        """
        Compute the median of the previously added data.

        Returns
        -------
        median_image
            2-dimensional array of shape [``imrows``, ``imcols``] containing
            median values computed across all images previously provided to ``add_image``.
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
