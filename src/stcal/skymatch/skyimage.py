"""
The ``skyimage`` module contains algorithms that are used by ``skymatch``.

Manage all of the information for footprints (image outlines)
on the sky as well as perform useful operations on these outlines such as
computing intersections and statistics in the overlap regions.
"""

import numpy as np
from gwcs import region
from spherical_geometry.polygon import SphericalPolygon

__all__ = ["SkyImage", "SkyGroup"]


class SkyImage:
    """
    Container that holds information about properties of a *single* image.

    Including:

    * image data;
    * WCS of the chip image;
    * bounding spherical polygon;
    * id;
    * pixel area;
    * sky background value;
    * sky statistics parameters;
    * mask associated image data indicating "good" (1) data.

    """

    def __init__(
        self,
        image,
        mask,
        wcs_fwd,
        wcs_inv,
        skystat,
        sky_id=None,
        stepsize=None,
        meta=None,
    ):
        """Initialize the SkyImage object.

        Parameters
        ----------
        image : numpy.ndarray
            A 2D array of image data.

        mask : numpy.ndarray
            A 2D array that indicates
            which pixels in the input `image` should be used for sky
            computations (``1``) and which pixels should **not** be used
            for sky computations (``0``).

        wcs_fwd : collections.abc.Callable
            "forward" pixel-to-world transformation function.

        wcs_inv : collections.abc.Callable
            "inverse" world-to-pixel transformation function.

        skystat : collections.abc.Callable, None, optional
            A callable object that takes a either a 2D image (2D
            `numpy.ndarray`) or a list of pixel values (a Nx1 array) and
            returns a tuple of two values: some statistics (e.g., mean,
            median, etc.) and number of pixels/values from the input image
            used in computing that statistics.

        sky_id : typing.Any
            The value of this parameter is simple stored within the `SkyImage`
            object. While it can be of any type, it is preferable that `id` be
            of a type with nice string representation.

        stepsize : int, None, optional
            Spacing between vertices of the image's bounding polygon. Default
            value of `None` creates bounding polygons with four vertices
            corresponding to the corners of the image.

        meta : dict, None, optional
            A dictionary of various items to be stored within the `SkyImage`
            object.

        """
        if image.shape != mask.shape:
            raise ValueError("'mask' must have the same shape as 'image'.")

        self.image = image
        self.mask = mask = np.asanyarray(mask, dtype=bool)

        self.meta = meta
        self.sky_id = sky_id

        # WCS
        self.wcs_fwd = wcs_fwd
        self.wcs_inv = wcs_inv

        # initial sky value:
        self.sky = 0.0
        self.is_sky_valid = False

        # create spherical polygon bounding the image
        self.calc_bounding_polygon(stepsize)

        # set sky statistics function (NOTE: it must return statistics and
        # the number of pixels used after clipping)
        self.skystat = skystat

    def intersection(self, skyimage):
        """
        Compute intersection of this `SkyImage`.

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
        # FIXME unused outside of this class
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
        """Compute image's bounding polygon.

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
        ny, nx = self.image.shape

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
        sl = np.s_[nptx : nptx + npty]
        borderx[sl] = nx - 0.5
        bordery[sl] = ys
        # "top"
        sl = np.s_[nptx + npty : 2 * nptx + npty]
        borderx[sl] = xs[::-1]
        bordery[sl] = ny - 0.5
        # "left"
        sl = np.s_[2 * nptx + npty : -1]
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

        self._polygon = SphericalPolygon.from_radec(ra, dec)
        self._poly_area = np.fabs(self._polygon.area())

    def calc_sky(self, overlap=None, delta=True):
        """
        Compute sky background value.

        Parameters
        ----------
        overlap : SkyImage, SkyGroup, None, optional
            This parameter is used to indicate that sky statistics
            should computed only in the region of intersection of *this*
            image with the `SkyImage` or `SkyGroup` indicated by `overlap`.
            When `overlap` is `None`, sky statistics will be computed over
            the entire image.

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
        if overlap is None:
            if self.mask is None:
                data = self.image
            else:
                data = self.image[self.mask]

            polyarea = self._poly_area

        else:
            fill_mask = np.zeros(self.image.shape, dtype=bool)

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

            if polyarea == 0.0:
                return None, 0, 0.0

            for ra, dec in radec:
                if len(ra) < 4:
                    continue

                # set pixels in 'fill_mask' that are inside a polygon to True:
                x, y = self.wcs_inv(ra, dec, with_bounding_box=False)
                poly_vert = list(zip(x, y, strict=True))

                polygon = region.Polygon(True, poly_vert)
                fill_mask = polygon.scan(fill_mask)

            if self.mask is not None:
                fill_mask &= self.mask

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


class SkyGroup:
    """
    Collection of :py:class:`SkyImage` objects.

    Holds multiple :py:class:`SkyImage` objects whose sky background values
    must be adjusted together.

    `SkyGroup` provides methods for obtaining bounding polygon of the group
    of :py:class:`SkyImage` objects and to compute sky value of the group.

    """

    def __init__(self, images, sky_id=None):
        self._images = images
        self._update_bounding_polygon()
        self.sky_id = sky_id
        self._sky = 0.0

    @property
    def sky(self):
        """Sky background value. See `calc_sky` for more details."""
        return self._sky

    @sky.setter
    def sky(self, sky):
        delta_sky = sky - self._sky
        self._sky = sky
        for im in self._images:
            im.sky += delta_sky

    @property
    def polygon(self):
        """Get image's bounding polygon."""
        # FIXME unused outside of this class
        return self._polygon

    def intersection(self, skyimage):
        """
        Compute intersection.

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
        # FIXME unused outside of this class
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
        polygons = [im._polygon for im in self._images]  # noqa: SLF001
        if len(polygons) == 0:
            self._polygon = SphericalPolygon([])
        else:
            self._polygon = SphericalPolygon.multi_union(polygons)

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
        yield from self._images

    def insert(self, idx, value):
        """Insert a `SkyImage` into the group."""
        if not isinstance(value, SkyImage):
            raise TypeError("Item must be of 'SkyImage' type")
        value.sky += self.sky
        self._images.insert(idx, value)
        self._update_bounding_polygon()

    def append(self, value):
        """Append a `SkyImage` to the group."""
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
        overlap : SkyImage, SkyGroup, None, optional
            This parameter is used to indicate that sky statistics
            should computed only in the region of intersection of *this*
            image with the `SkyImage` or `SkyGroup` indicated by `overlap`.
            When `overlap` is `None`, sky statistics will be computed over the
            entire image.

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
                wsky += sky * npix
                wght += npix

        if wght == 0.0 or area == 0.0:
            return None, wght, area
        else:
            return wsky / wght, wght, area
