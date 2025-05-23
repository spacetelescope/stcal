"""
The ``skyimage`` module contains algorithms that are used by
``skymatch`` to manage all of the information for footprints (image outlines)
on the sky as well as perform useful operations on these outlines such as
computing intersections and statistics in the overlap regions.

:Authors: Mihai Cara (contact: help@stsci.edu)


"""

# STDLIB
import abc
import tempfile

# THIRD-PARTY
import numpy as np
from gwcs import region
from spherical_geometry.polygon import SphericalPolygon

# LOCAL
from . skystatistics import SkyStats


__all__ = ['SkyImage', 'SkyGroup', 'DataAccessor', 'NDArrayInMemoryAccessor',
           'NDArrayMappedAccessor']


class DataAccessor(abc.ABC):
    """ Base class for all data accessors. Provides a common interface to
        access data.
    """
    @abc.abstractmethod
    def get_data(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def set_data(self, data):  # pragma: no cover
        """ Sets data.

        Parameters
        ----------
        data : numpy.ndarray
            Data array to be set.

        """
        pass

    @abc.abstractmethod
    def get_data_shape(self):  # pragma: no cover
        pass


class NDArrayInMemoryAccessor(DataAccessor):
    """ Accessor for in-memory `numpy.ndarray` data. """
    def __init__(self, data):
        super().__init__()
        self._data = data

    def get_data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    def get_data_shape(self):
        return np.shape(self._data)


class NDArrayMappedAccessor(DataAccessor):
    """ Data accessor for arrays stored in temporary files. """
    def __init__(self, data, tmpfile=None, prefix='tmp_skymatch_',
                 suffix='.npy', tmpdir=''):
        super().__init__()
        if tmpfile is None:
            self._close = True
            self._tmp = tempfile.NamedTemporaryFile(
                prefix=prefix,
                suffix=suffix,
                dir=tmpdir
            )
            if not self._tmp:
                raise RuntimeError("Unable to create temporary file.")
        else:
            # temp file managed by the caller
            self._close = False
            self._tmp = tmpfile

        self.set_data(data)

    def get_data(self):
        self._tmp.seek(0)
        return np.load(self._tmp)

    def set_data(self, data):
        data = np.asanyarray(data)
        self._data_shape = data.shape
        self._tmp.seek(0)
        np.save(self._tmp, data)

    def __del__(self):
        if self._close:
            self._tmp.close()

    def get_data_shape(self):
        return self._data_shape


class SkyImage:
    """
    Container that holds information about properties of a *single*
    image such as:

    * image data;
    * WCS of the chip image;
    * bounding spherical polygon;
    * id;
    * pixel area;
    * sky background value;
    * sky statistics parameters;
    * mask associated image data indicating "good" (1) data.

    """

    def __init__(self, image, wcs_fwd, wcs_inv, pix_area=1.0, convf=1.0,
                 mask=None, sky_id=None, skystat=None, stepsize=None, meta=None,
                 reduce_memory_usage=True):
        """ Initializes the SkyImage object.

        Parameters
        ----------
        image : numpy.ndarray, `NDArrayDataAccessor`
            A 2D array of image data or a `NDArrayDataAccessor`.

        wcs_fwd : collections.abc.Callable
            "forward" pixel-to-world transformation function.

        wcs_inv : collections.abc.Callable
            "inverse" world-to-pixel transformation function.

        pix_area : float, optional
            Average pixel's sky area.

        convf : float, optional
            Conversion factor that when multiplied to `image` data converts
            the data to "uniform" (across multiple images) surface
            brightness units.

            .. note::

              The functionality to support this conversion is not yet
              implemented and at this moment `convf` is ignored.

        mask : numpy.ndarray, `NDArrayDataAccessor`
            A 2D array or `NDArrayDataAccessor` of a 2D array that indicates
            which pixels in the input `image` should be used for sky
            computations (``1``) and which pixels should **not** be used
            for sky computations (``0``).

        sky_id : typing.Any
            The value of this parameter is simple stored within the `SkyImage`
            object. While it can be of any type, it is preferable that `id` be
            of a type with nice string representation.

        skystat : collections.abc.Callable, None, optional
            A callable object that takes a either a 2D image (2D
            `numpy.ndarray`) or a list of pixel values (a Nx1 array) and
            returns a tuple of two values: some statistics (e.g., mean,
            median, etc.) and number of pixels/values from the input image
            used in computing that statistics.

            When `skystat` is not set, `SkyImage` will use
            :py:class:`~stcal.skymatch.skystatistics.SkyStats` object
            to perform sky statistics on image data.

        stepsize : int, None, optional
            Spacing between vertices of the image's bounding polygon. Default
            value of `None` creates bounding polygons with four vertices
            corresponding to the corners of the image.

        meta : dict, None, optional
            A dictionary of various items to be stored within the `SkyImage`
            object.

        reduce_memory_usage : bool, optional
            Indicates whether to attempt to minimize memory usage by attaching
            input ``image`` and/or ``mask`` `numpy.ndarray` arrays to
            file-mapped accessor. This has no effect when input parameters
            ``image`` and/or ``mask`` are already of `NDArrayDataAccessor`
            objects.

        """
        self._image = None
        self._mask = None
        self._image_shape = None
        self._mask_shape = None
        self._reduce_memory_usage = reduce_memory_usage

        self.image = image

        self.convf = convf
        self.meta = meta
        self.sky_id = sky_id
        self.pix_area = pix_area

        # WCS
        self.wcs_fwd = wcs_fwd
        self.wcs_inv = wcs_inv

        # initial sky value:
        self.sky = 0.0
        self.sky_is_valid = False

        self.mask = mask

        # create spherical polygon bounding the image
        if image is None or wcs_fwd is None or wcs_inv is None:
            self._radec = [(np.array([]), np.array([]))]
            self._polygon = SphericalPolygon([])
            self._poly_area = 0.0

        else:
            self.calc_bounding_polygon(stepsize)

        # set sky statistics function (NOTE: it must return statistics and
        # the number of pixels used after clipping)
        if skystat is None:
            self.set_builtin_skystat()
        else:
            self.skystat = skystat

    @property
    def mask(self):
        """ Set or get `SkyImage`'s ``mask`` data array or `None`. """
        if self._mask is None:
            return None
        else:
            return self._mask.get_data()

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = None
            self._mask_shape = None

        elif isinstance(mask, DataAccessor):
            if self._image is None:
                raise ValueError("'mask' must be None when 'image' is None")

            self._mask = mask
            self._mask_shape = mask.get_data_shape()

            # check that mask has the same shape as image:
            if self._mask_shape != self.image_shape:
                raise ValueError("'mask' must have the same shape as 'image'.")

        else:
            if self._image is None:
                raise ValueError("'mask' must be None when 'image' is None")

            mask = np.asanyarray(mask, dtype=bool)
            self._mask_shape = mask.shape

            # check that mask has the same shape as image:
            if self._mask_shape != self.image_shape:
                raise ValueError("'mask' must have the same shape as 'image'.")

            if self._mask is None:
                if self._reduce_memory_usage:
                    self._mask = NDArrayMappedAccessor(
                        mask,
                        prefix='tmp_skymatch_mask_'
                    )
                else:
                    self._mask = NDArrayInMemoryAccessor(mask)
            else:
                self._mask.set_data(mask)

    @property
    def image(self):
        """ Set or get `SkyImage`'s ``image`` data array. """
        if self._image is None:
            return None
        else:
            return self._image.get_data()

    @image.setter
    def image(self, image):
        if image is None:
            self._image = None
            self._image_shape = None
            self.mask = None

        if isinstance(image, DataAccessor):
            self._image = image
            self._image_shape = image.get_data_shape()

        else:
            image = np.asanyarray(image)
            self._image_shape = image.shape
            if self._image is None:
                if self._reduce_memory_usage:
                    self._image = NDArrayMappedAccessor(
                        image,
                        prefix='tmp_skymatch_image_'
                    )
                else:
                    self._image = NDArrayInMemoryAccessor(image)
            else:
                self._image.set_data(image)

    @property
    def image_shape(self):
        """ Get `SkyImage`'s ``image`` data shape. """
        if self._image_shape is None and self._image is not None:
            self._image_shape = self._image.get_data_shape()
        return self._image_shape

    @property
    def poly_area(self):
        """ Get bounding polygon area in srad units.
        """
        return self._poly_area

    @property
    def radec(self):
        """
        Get RA and DEC of the vertices of the bounding polygon as a
        `~numpy.ndarray` of shape (N, 2) where N is the number of vertices + 1.
        """
        return self._radec

    @property
    def polygon(self):
        """ Get image's bounding polygon.
        """
        return self._polygon

    def intersection(self, skyimage):
        """
        Compute intersection of this `SkyImage` object and another
        `SkyImage`, `SkyGroup`, or
        :py:class:`~spherical_geometry.polygon.SphericalPolygon`
        object.

        Parameters
        ----------
        skyimage : SkyImage, SkyGroup, spherical_geometry.polygon.SphericalPolygon
            Another object that should be intersected with this `SkyImage`.

        Returns
        -------
        polygon : `SphericalPolygon`
            A :py:class:`~spherical_geometry.polygon.SphericalPolygon` that is
            the intersection of this `SkyImage` and `skyimage`.

        """
        if isinstance(skyimage, (SkyImage, SkyGroup)):
            other = skyimage.polygon
        else:
            other = skyimage

        pts1 = np.sort(list(self._polygon.points)[0], axis=0)
        pts2 = np.sort(list(other.points)[0], axis=0)
        if np.allclose(pts1, pts2, rtol=0, atol=5e-9):
            intersect_poly = self._polygon.copy()
        else:
            intersect_poly = self._polygon.intersection(other)
        return intersect_poly

    def calc_bounding_polygon(self, stepsize=None):
        """ Compute image's bounding polygon.

        Parameters
        ----------
        stepsize : int, None, optional
            Indicates the maximum separation between two adjacent vertices
            of the bounding polygon along each side of the image. Corners
            of the image are included automatically. If `stepsize` is `None`,
            bounding polygon will contain only vertices of the image.

        Notes
        -----
        The bounding polygon is defined from corners of pixels whereas the pixel
        coordinates refer to their centers and therefore the lower-left corner
        is located at (-0.5, -0.5)
        """
        ny, nx = self.image_shape

        if stepsize is None:
            nint_x = 2
            nint_y = 2
        else:
            nint_x = max(2, int(np.ceil((nx + 1.0) / stepsize)))
            nint_y = max(2, int(np.ceil((ny + 1.0) / stepsize)))

        xs = np.linspace(-0.5, nx - 0.5, nint_x, dtype=float)
        ys = np.linspace(-0.5, ny - 0.5, nint_y, dtype=float)[1:-1]
        nptx = xs.size
        npty = ys.size

        npts = 2 * (nptx + npty)

        borderx = np.empty((npts + 1,), dtype=float)
        bordery = np.empty((npts + 1,), dtype=float)

        # "bottom" points:
        borderx[:nptx] = xs
        bordery[:nptx] = -0.5
        # "right"
        sl = np.s_[nptx:nptx + npty]
        borderx[sl] = nx - 0.5
        bordery[sl] = ys
        # "top"
        sl = np.s_[nptx + npty:2 * nptx + npty]
        borderx[sl] = xs[::-1]
        bordery[sl] = ny - 0.5
        # "left"
        sl = np.s_[2 * nptx + npty:-1]
        borderx[sl] = -0.5
        bordery[sl] = ys[::-1]

        # close polygon:
        borderx[-1] = borderx[0]
        bordery[-1] = bordery[0]

        ra, dec = self.wcs_fwd(borderx, bordery, with_bounding_box=False)
        # TODO: for strange reasons, occasionally ra[0] != ra[-1] and/or
        #       dec[0] != dec[-1] (even though we close the polygon in the
        #       previous two lines). Then SphericalPolygon fails because
        #       points are not closed. Therefore we force it to be closed:
        ra[-1] = ra[0]
        dec[-1] = dec[0]

        self._radec = [(ra, dec)]
        self._polygon = SphericalPolygon.from_radec(ra, dec)
        self._poly_area = np.fabs(self._polygon.area())

    def set_builtin_skystat(self, skystat='median', lower=None, upper=None,
                            nclip=5, lsigma=4.0, usigma=4.0, binwidth=0.1):
        """
        Replace already set `skystat` with a "built-in" version of a
        statistics callable object used to measure sky background.

        See :py:class:`~stcal.skymatch.skystatistics.SkyStats` for the
        parameter description.

        """
        self.skystat = SkyStats(
            skystat=skystat,
            lower=lower,
            upper=upper,
            nclip=nclip,
            lsig=lsigma,
            usig=usigma,
            binwidth=binwidth
        )

    def calc_sky(self, overlap=None, delta=True):
        """
        Compute sky background value.

        Parameters
        ----------
        overlap : SkyImage, SkyGroup, `SphericalPolygon`, list[tuple[typing.Any]], None, optional
            Another `SkyImage`, `SkyGroup`,
            :py:class:`spherical_geometry.polygon.SphericalPolygon`, or
            a list of tuples of (RA, DEC) of vertices of a spherical
            polygon. This parameter is used to indicate that sky statistics
            should computed only in the region of intersection of *this*
            image with the polygon indicated by `overlap`. When `overlap` is
            `None`, sky statistics will be computed over the entire image.

        delta : bool, optional
            Should this function return absolute sky value or the difference
            between the computed value and the value of the sky stored in the
            `sky` property.

        Returns
        -------
        skyval : float, None
            Computed sky value (absolute or relative to the `sky` attribute).
            If there are no valid data to perform this computations (e.g.,
            because this image does not overlap with the image indicated by
            `overlap`), `skyval` will be set to `None`.

        npix : int
            Number of pixels used to compute sky statistics.

        polyarea : float
            Area (in srad) of the polygon that bounds data used to compute
            sky statistics.
    
        Notes
        -----
        Due to a bug in the sphere package, see
        https://github.com/spacetelescope/sphere/issues/74
        intersections with polygons formed as union does not work.
        For this reason I re-implement 'calc_sky' below with a workaround for
        the bug. The original implementation should be used when the bug is
        fixed.
        """
        if overlap is None:

            if self._mask is None:
                data = self.image
            else:
                data = self.image[self._mask.get_data()]

            polyarea = self.poly_area

        else:
            fill_mask = np.zeros(self.image_shape, dtype=bool)

            if isinstance(overlap, SkyImage):
                intersection = self.intersection(overlap)
                polyarea = np.fabs(intersection.area())
                radec = list(intersection.to_radec())

            elif isinstance(overlap, SkyGroup):
                radec = []
                polyarea = 0.0
                for im in overlap:
                    intersection = self.intersection(im)
                    polyarea1 = np.fabs(intersection.area())
                    if polyarea1 == 0.0:
                        continue
                    polyarea += polyarea1
                    radec += list(intersection.to_radec())

            elif isinstance(overlap, SphericalPolygon):
                radec = []
                polyarea = 0.0
                for p in overlap._polygons:
                    intersection = self.intersection(SphericalPolygon([p]))
                    polyarea1 = np.fabs(intersection.area())
                    if polyarea1 == 0.0:
                        continue
                    polyarea += polyarea1
                    radec += list(intersection.to_radec())

            else:  # assume a list of (ra, dec) tuples:
                radec = []
                polyarea = 0.0
                for r, d in overlap:
                    poly = SphericalPolygon.from_radec(r, d)
                    polyarea1 = np.fabs(poly.area())
                    if polyarea1 == 0.0 or len(r) < 4:
                        continue
                    polyarea += polyarea1
                    radec.append(self.intersection(poly).to_radec())

            if polyarea == 0.0:
                return None, 0, 0.0

            for ra, dec in radec:
                if len(ra) < 4:
                    continue

                # set pixels in 'fill_mask' that are inside a polygon to True:
                x, y = self.wcs_inv(ra, dec, with_bounding_box=False)
                poly_vert = list(zip(*[x, y]))

                polygon = region.Polygon(True, poly_vert)
                fill_mask = polygon.scan(fill_mask)

            if self._mask is not None:
                fill_mask &= self._mask.get_data()

            data = self.image[fill_mask]

            if data.size < 1:
                return None, 0, 0.0

        # Calculate sky
        try:
            skyval, npix = self.skystat(data)
        except ValueError:
            return None, 0, 0.0

        if not np.isfinite(skyval):
            return None, 0, 0.0

        if delta:
            skyval -= self.sky

        return skyval, npix, polyarea

    def copy(self):
        """
        Return a shallow copy of the `SkyImage` object.
        """
        si = SkyImage(
            image=None,
            wcs_fwd=self.wcs_fwd,
            wcs_inv=self.wcs_inv,
            pix_area=self.pix_area,
            convf=self.convf,
            mask=None,
            sky_id=self.sky_id,
            stepsize=None,
            meta=self.meta
        )

        si._image = self._image
        si._mask = self._mask
        si._image_shape = self._image_shape
        si._mask_shape = self._mask_shape
        si._reduce_memory_usage = self._reduce_memory_usage

        si._radec = self._radec
        si._polygon = self._polygon
        si._poly_area = self._poly_area
        si.sky = self.sky
        return si


class SkyGroup:
    """
    Holds multiple :py:class:`SkyImage` objects whose sky background values
    must be adjusted together.

    `SkyGroup` provides methods for obtaining bounding polygon of the group
    of :py:class:`SkyImage` objects and to compute sky value of the group.

    """

    def __init__(self, images, sky_id=None, sky=0.0):

        if isinstance(images, SkyImage):
            self._images = [images]

        elif hasattr(images, '__iter__'):
            self._images = []
            for im in images:
                if not isinstance(im, SkyImage):
                    raise TypeError("Each element of the 'images' parameter "
                                    "must be an 'SkyImage' object.")
                self._images.append(im)

        else:
            raise TypeError(
                "Parameter 'images' must be either a single 'SkyImage' object "
                "or a list of 'SkyImage' objects"
            )

        self.sky_id = sky_id
        self._update_bounding_polygon()
        self._sky = sky
        for im in self._images:
            im.sky += sky

    @property
    def sky(self):
        """ Sky background value. See `calc_sky` for more details.
        """
        return self._sky

    @sky.setter
    def sky(self, sky):
        delta_sky = sky - self._sky
        self._sky = sky
        for im in self._images:
            im.sky += delta_sky

    @property
    def radec(self):
        """
        Get RA and DEC of the vertices of the bounding polygon as a
        `~numpy.ndarray` of shape (N, 2) where N is the number of vertices + 1.

        """
        return self._radec

    @property
    def polygon(self):
        """ Get image's bounding polygon.
        """
        return self._polygon

    def intersection(self, skyimage):
        """
        Compute intersection of this `SkyImage` object and another
        `SkyImage`, `SkyGroup`, or
        :py:class:`~spherical_geometry.polygon.SphericalPolygon`
        object.

        Parameters
        ----------
        skyimage : SkyImage, SkyGroup, `SphericalPolygon`
            Another object that should be intersected with this `SkyImage`.

        Returns
        -------
        intersect_poly : `SphericalPolygon`
            A :py:class:`~spherical_geometry.polygon.SphericalPolygon` that is
            the intersection of this `SkyImage` and `skyimage`.

        """
        if isinstance(skyimage, (SkyImage, SkyGroup)):
            other = skyimage.polygon
        else:
            other = skyimage

        pts1 = np.sort(list(self._polygon.points)[0], axis=0)
        pts2 = np.sort(list(other.points)[0], axis=0)
        if np.allclose(pts1, pts2, rtol=0, atol=1e-8):
            intersect_poly = self._polygon.copy()
        else:
            intersect_poly = self._polygon.intersection(other)
        return intersect_poly

    def _update_bounding_polygon(self):
        polygons = [im.polygon for im in self._images]
        if len(polygons) == 0:
            self._polygon = SphericalPolygon([])
            self._radec = []
        else:
            self._polygon = SphericalPolygon.multi_union(polygons)
            self._radec = list(self._polygon.to_radec())

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        return self._images[idx]

    def __setitem__(self, idx, value):
        if not isinstance(value, SkyImage):
            raise TypeError("Item must be of 'SkyImage' type")
        value.sky += self.sky
        self._images[idx] = value
        self._update_bounding_polygon()

    def __delitem__(self, idx):
        del self._images[idx]
        if len(self._images) == 0:
            self._sky = 0.0
            self.sky_id = None
        self._update_bounding_polygon()

    def __iter__(self):
        for image in self._images:
            yield image

    def insert(self, idx, value):
        """Inserts a `SkyImage` into the group.
        """
        if not isinstance(value, SkyImage):
            raise TypeError("Item must be of 'SkyImage' type")
        value.sky += self.sky
        self._images.insert(idx, value)
        self._update_bounding_polygon()

    def append(self, value):
        """Appends a `SkyImage` to the group.
        """
        if not isinstance(value, SkyImage):
            raise TypeError("Item must be of 'SkyImage' type")
        value.sky += self.sky
        self._images.append(value)
        self._update_bounding_polygon()

    def calc_sky(self, overlap=None, delta=True):
        """
        Compute sky background value.

        Parameters
        ----------
        overlap : SkyImage, SkyGroup, `SphericalPolygon`, list[tuple[typing.Any]], None, optional
            Another `SkyImage`, `SkyGroup`,
            :py:class:`spherical_geometry.polygon.SphericalPolygon`, or
            a list of tuples of (RA, DEC) of vertices of a spherical
            polygon. This parameter is used to indicate that sky statistics
            should computed only in the region of intersection of *this*
            image with the polygon indicated by `overlap`. When `overlap` is
            `None`, sky statistics will be computed over the entire image.

        delta : bool, optional
            Should this function return absolute sky value or the difference
            between the computed value and the value of the sky stored in the
            `sky` property.

        Returns
        -------
        skyval : float, None
            Computed sky value (absolute or relative to the `sky` attribute).
            If there are no valid data to perform this computations (e.g.,
            because this image does not overlap with the image indicated by
            `overlap`), `skyval` will be set to `None`.

        npix : int
            Number of pixels used to compute sky statistics.

        polyarea : float
            Area (in srad) of the polygon that bounds data used to compute
            sky statistics.

        """

        if len(self._images) == 0:
            return None, 0, 0.0

        wght = 0
        area = 0.0

        if overlap is None:
            # compute minimum sky across all images in the group:
            wsky = None

            for image in self._images:
                # make sure all images have the same background:
                image.background = self.sky

                sky, npix, imarea = image.calc_sky(overlap=None, delta=delta)

                if sky is None:
                    continue

                if wsky is None or wsky > sky:
                    wsky = sky
                    wght = npix
                    area = imarea

            return wsky, wght, area

        # compute weighted sky in various overlaps:
        wsky = 0.0

        for image in self._images:
            # make sure all images have the same background:
            image.background = self.sky

            sky, npix, area1 = image.calc_sky(overlap=overlap, delta=delta)

            area += area1

            if sky is not None and npix > 0:
                pix_area = npix * image.pix_area
                wsky += sky * pix_area
                wght += pix_area

        if wght == 0.0 or area == 0.0:
            return None, wght, area
        else:
            return wsky / wght, wght, area
