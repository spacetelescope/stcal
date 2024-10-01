import os
from pathlib import Path

import numpy as np
import pytest

from stcal.outlier_detection.median import (
    MedianComputer,
    _DiskAppendableArray,
    _OnDiskMedian,
    nanmedian3D,
)


def test_disk_appendable_array(tmpdir):

    slice_shape = (8, 7)
    dtype = "float32"
    tempdir = Path(tmpdir) / Path("tmptest")
    Path.mkdir(tempdir)
    fname = tempdir / "test.bin"

    arr = _DiskAppendableArray(slice_shape, dtype, fname)

    # check temporary file setup
    assert str(arr._filename).split("/")[-1] in os.listdir(tempdir)  # noqa: SLF001
    assert len(os.listdir(tempdir)) == 1
    assert arr.shape == (0, *slice_shape)

    # check cwd contains no files
    assert all(not Path.is_file(Path(f)) for f in os.listdir(tmpdir))

    # check expected append failures
    candidate = np.zeros((7, 7), dtype=dtype)
    with pytest.raises(ValueError, match="shape"):
        arr.append(candidate)
    candidate = np.zeros((8, 7), dtype="float64")
    with pytest.raises(ValueError, match="dtype"):
        arr.append(candidate)

    # check append and read
    candidate0 = np.zeros(slice_shape, dtype=dtype)
    candidate1 = np.full(slice_shape, 2, dtype=dtype)
    candidate2 = np.full(slice_shape, np.nan, dtype=dtype)
    for candidate in [candidate0, candidate1, candidate2]:
        arr.append(candidate)

    arr_in_memory = arr.read()

    assert arr_in_memory.shape == (3, *slice_shape)
    assert np.all(arr_in_memory[0] == candidate0)
    assert np.all(arr_in_memory[1] == candidate1)
    assert np.allclose(arr_in_memory[2], candidate2, equal_nan=True)


def test_disk_appendable_array_bad_inputs(tmpdir):

    slice_shape = (8, 7)
    dtype = "float32"
    tempdir = Path(tmpdir) / Path("tmptest")
    fname = "test.bin"

    # test input directory does not exist
    with pytest.raises(FileNotFoundError):
        _DiskAppendableArray(slice_shape, dtype, tempdir / fname)

    # make the input directory
    Path.mkdir(tempdir)

    # ensure failure if slice_shape is not 2-D
    with pytest.raises(ValueError, match="slice shape"):
        _DiskAppendableArray((3, 5, 7), dtype, tempdir / fname)

    # ensure failure if dtype is not valid
    with pytest.raises(TypeError):
        _DiskAppendableArray(slice_shape, "float3", tempdir / fname)

    # ensure failure if pass directory instead of filename
    with pytest.raises(IsADirectoryError):
        _DiskAppendableArray(slice_shape, "float3", tempdir)


def test_on_disk_median(tmpdir):

    library_length = 3
    frame_shape = (21, 20)
    dtype = "float32"
    tempdir = Path(tmpdir) / Path("tmptest")
    Path.mkdir(tempdir)
    shape = (library_length, *frame_shape)

    median_computer = _OnDiskMedian(shape, dtype=dtype, tempdir=tempdir)

    # test compute buffer indices
    # buffer size equals size of single input model by default
    # which means we expect same number of sections as library length
    # in reality there is often one more section than that because
    # need integer number of rows per section, but math is exact in this case
    expected_buffer_size = frame_shape[0] * frame_shape[1] * \
        np.dtype(dtype).itemsize
    expected_section_nrows = frame_shape[0] // library_length
    assert median_computer.nsections == library_length
    assert median_computer.section_nrows == expected_section_nrows
    assert median_computer.buffer_size == expected_buffer_size

    # test temp file setup
    assert len(os.listdir(tempdir)) == 1
    assert str(median_computer._temp_path)\
        .startswith(str(tempdir))  # noqa: SLF001
    assert len(os.listdir(median_computer._temp_path)) \
        == library_length  # noqa: SLF001
    # check cwd and parent tempdir contain no files
    assert all(not Path.is_file(Path(f)) for f in os.listdir(tmpdir))
    assert all(not Path.is_file(Path(f)) for f in os.listdir(tempdir))

    # test validate data
    candidate = np.zeros((20, 20), dtype=dtype)
    with pytest.raises(ValueError, match="shape"):
        median_computer.add_image(candidate)
    candidate = np.zeros((21, 20), dtype="float64")
    with pytest.raises(ValueError, match="dtype"):
        median_computer.add_image(candidate)

    # test add and compute
    candidate0 = np.full(frame_shape, 3, dtype=dtype)
    candidate1 = np.full(frame_shape, 2, dtype=dtype)
    candidate2 = np.full(frame_shape, np.nan, dtype=dtype)
    for candidate in [candidate0, candidate1, candidate2]:
        median_computer.add_image(candidate)
    median = median_computer.compute_median()
    median_computer.cleanup()
    assert median.shape == frame_shape
    assert np.all(median == 2.5)

    # test expected error trying to add too many frames
    # for loop to ensure always happens, not just the first time
    candidate3 = np.zeros_like(candidate0)
    for _ in range(2):
        with pytest.raises(IndexError):
            median_computer.add_image(candidate3)

    # test cleanup of tmpdir and everything else
    assert not Path.exists(median_computer._temp_path)  # noqa: SLF001
    assert len(os.listdir(tempdir)) == 0


def test_computer():
    """Ensure MedianComputer works the same on disk and in memory"""
    full_shape = (3, 21, 20)
    comp_memory = MedianComputer(full_shape, True)
    comp_disk = MedianComputer(full_shape, False)
    for i in range(full_shape[0]):
        frame = np.full((21, 20), i, dtype=np.float32)
        comp_memory.append(frame, i)
        comp_disk.append(frame, i)
    assert np.allclose(comp_memory.evaluate(), comp_disk.evaluate())


def test_on_disk_median_bad_inputs(tmpdir):

    library_length = 3
    frame_shape = (21, 20)
    dtype = "float32"
    tempdir = Path(tmpdir) / Path("tmptest")
    Path.mkdir(tempdir)
    shape = (library_length, *frame_shape)

    with pytest.raises(ValueError, match="shape"):
        _OnDiskMedian(frame_shape, dtype=dtype, tempdir=tempdir)

    with pytest.raises(TypeError):
        _OnDiskMedian(shape, dtype="float3", tempdir=tempdir)

    with pytest.raises(FileNotFoundError):
        _OnDiskMedian(shape, dtype="float32", tempdir="dne")

    # ensure unreasonable buffer size will get set to minimum reasonable buffer
    min_buffer = np.dtype(dtype).itemsize*frame_shape[1]*library_length
    median_computer = _OnDiskMedian(shape,
                                   dtype=dtype,
                                   tempdir=tempdir,
                                   buffer_size=-1)
    assert median_computer.buffer_size == min_buffer
    median_computer.cleanup()


def test_nanmedian3D():

    shp = (11, 50, 60)
    generator = np.random.default_rng(77)
    cube = generator.normal(size=shp)
    cube[5, 5:7, 5:8] = np.nan
    med = nanmedian3D(cube.astype(np.float32))

    assert med.dtype == np.float32
    assert np.allclose(med, np.nanmedian(cube, axis=0), equal_nan=True)
