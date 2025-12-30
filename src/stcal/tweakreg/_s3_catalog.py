import concurrent.futures
import functools

import numpy as np
import pyarrow.csv as csv
import pyarrow.fs
import pyarrow.parquet as pq
from astropy import coordinates
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from astropy_healpix import HEALPix

__all__ = []


S3_NAMES_TO_CATALOGS = {
    "GAIADR3_S3": "s3://stpubdata/gaia/gaia_dr3/public/hats/gaia/",
}
S3_CATALOGS = list(S3_NAMES_TO_CATALOGS.keys())
SPATIAL_INDEX_COLUMN = "_healpix_29"
COL_NAMES = (
    "ra",
    "ra_error",
    "dec",
    "dec_error",
    "parallax",
    "parallax_error",
    "source_id",  # objID
    "ref_epoch",  # epoch
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "phot_g_mean_mag",  # mag
)


def _correct_for_proper_motion(catalog, epoch):
    # remove rows without pmra, pmdec, parallax
    catalog = catalog[~(catalog["pmra"].mask | catalog["pmdec"].mask | catalog["parallax"].mask)]
    dt = epoch - catalog["ref_epoch"]
    unitspherical = coordinates.UnitSphericalRepresentation(
        catalog["ra"] * u.deg,
        catalog["dec"] * u.deg,
    )
    xyz = unitspherical.to_cartesian().xyz
    unit_vectors = unitspherical.unit_vectors()
    rahat = unit_vectors["lon"].xyz
    dechat = unit_vectors["lat"].xyz
    earthcoord = coordinates.get_body_barycentric("earth", Time(epoch, format="decimalyear"))
    earthcoord = earthcoord.xyz.to(u.AU).value
    radpermas = np.pi / (180 * 3600 * 1000)
    newxyz = xyz + rahat * dt * radpermas * catalog["pmra"] + dechat * dt * radpermas * catalog["pmdec"]
    plx = catalog["parallax"]
    newxyz -= (
        rahat * earthcoord.dot(rahat) * plx * radpermas + dechat * earthcoord.dot(dechat) * plx * radpermas
    )
    # stars move in the opposite direction of the earth -> minus sign
    newunitspherical = coordinates.UnitSphericalRepresentation.from_cartesian(
        coordinates.CartesianRepresentation(newxyz)
    )
    newra = newunitspherical.lon
    newdec = newunitspherical.lat
    catalog["ra"] = newra.to(u.deg).value
    catalog["dec"] = newdec.to(u.deg).value
    return catalog


@functools.lru_cache
def _get_partition_info(uri):
    # open the filesystem here to allow this to be cached (FileSystem instances aren't hashable)
    fs, fs_path = pyarrow.fs.FileSystem.from_uri(uri)
    csv_file = csv.read_csv(fs.open_input_file(fs_path + "/partition_info.csv"))
    return fs, fs_path, np.vstack((csv_file["Norder"].to_numpy(), csv_file["Npix"].to_numpy())).T


def _cone_search(ra, dec, search_radius, depth):
    hp = HEALPix(nside=2**depth, order="nested")
    return hp.cone_search_lonlat(ra * u.deg, dec * u.deg, search_radius * u.deg)


def _generate_depth_29_ranges(pixels, depth):
    depth_array = np.full(pixels.shape, depth)
    return np.vstack((pixels << ((29 - depth_array) * 2), (pixels + 1) << ((29 - depth_array) * 2))).T


def _find_pixels_in_catalog(catalog_pixels, search_pixels, search_depth):
    min_depth = catalog_pixels[:, 0].min()
    if search_depth < min_depth:
        raise ValueError(f"Invalid search_depth {search_depth} < {min_depth}")
    pixels_by_depth = {}
    for d, i in catalog_pixels.tolist():
        if d not in pixels_by_depth:
            pixels_by_depth[d] = set()
        pixels_by_depth[d].add(i)
    matching_pixels = set()
    for i in search_pixels:
        depth = search_depth
        while depth >= min_depth:
            if i in pixels_by_depth.get(depth, set()):
                matching_pixels.add((depth, i))
                break
            depth = depth - 1
            i = i >> 2
    return list(matching_pixels)


def _pixels_to_paths(pixels, fs_path):
    paths = []
    for pixel in pixels:
        depth, number = pixel
        directory = int(number / 10000) * 10000
        paths.append(f"{fs_path}/dataset/Norder={depth}/Dir={directory}/Npix={number}.parquet")
    return paths


def _depth_29_ranges_to_filters(dr):
    filters = []
    for hpx_range in dr:
        filters.append(
            [(SPATIAL_INDEX_COLUMN, ">=", hpx_range[0]), (SPATIAL_INDEX_COLUMN, "<", hpx_range[1])]
        )
    return filters


def _read_table(paths, fs, filters, columns):
    ds = pq.ParquetDataset(paths, filesystem=fs, filters=filters).read(columns=columns)
    table = Table({name: ds[name].to_numpy() for name in ds.column_names}, masked=True)
    # mask na values
    for colname in table.colnames:
        table[colname].mask = np.isnan(table[colname].data.data)
    return table


def _filter_table(table, ra, dec, search_radius):
    ra_rad = np.radians(table["ra"])
    dec_rad = np.radians(table["dec"])
    ra0 = np.radians(ra)
    dec0 = np.radians(dec)

    cos_angle = np.sin(dec_rad) * np.sin(dec0) + np.cos(dec_rad) * np.cos(dec0) * np.cos(ra_rad - ra0)

    # Clamp to valid range to avoid numerical issues
    cos_separation = np.clip(cos_angle, -1.0, 1.0)

    cos_radius = np.cos(np.radians(search_radius))
    return table[cos_separation >= cos_radius]


def _get_hats_sources(gaia_dr3_uri, ra, dec, search_radius, epoch=None, columns=None):
    # get the partition_info file: Norder, Npix
    fs, fs_path, hats_pixels = _get_partition_info(gaia_dr3_uri)
    max_depth = hats_pixels[:, 0].max()

    # perform cone search at max depth
    ipix = _cone_search(ra, dec, search_radius, max_depth)

    # generate depth 29 ranges for cone search
    dr = _generate_depth_29_ranges(ipix, max_depth)

    # find matching pixels in catalog
    pixels = _find_pixels_in_catalog(hats_pixels, ipix, max_depth)

    # convert pixels to paths
    paths = _pixels_to_paths(pixels, fs_path)

    # convert depth 29 ranges to pyarrow filters
    filters = _depth_29_ranges_to_filters(dr)

    # read and combine data
    table = _read_table(paths, fs, filters, columns)

    # do the final, accurate cone search
    return _filter_table(table, ra, dec, search_radius)


def get_s3_catalog(
    right_ascension,
    declination,
    epoch=None,
    search_radius=0.1,
    catalog="GAIADR3_S3",
    timeout=600,
):
    """Extract catalog from S3 web service.

    Parameters
    ----------
    right_ascension : float
        Right Ascension (RA) of center of field-of-view (in decimal degrees)

    declination : float
        Declination (Dec) of center of field-of-view (in decimal degrees)

    epoch : float, optional
        Reference epoch used to update the coordinates for proper motion
        (in decimal year). Default: None

    search_radius : float, optional
        Search radius (in decimal degrees) from field-of-view center to use
        for sources from catalog.  Default: 0.1 degrees

    catalog : str, optional
        Name of catalog to query, as defined by web-service. Default: 'GAIADR3_S3'

    timeout : float, optional
        Timeout in seconds to wait for the catalog web service to respond. Default: 600.0 s

    Returns
    -------
    `~astropy.table.Table`
        Table of returned sources with all columns as provided by catalog
    """
    s3_url = S3_NAMES_TO_CATALOGS[catalog]

    # use a thread to allow setting a timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _get_hats_sources, s3_url, right_ascension, declination, search_radius, epoch, list(COL_NAMES)
        )
        table = future.result(timeout=timeout)

    if epoch:
        table = _correct_for_proper_motion(table, epoch)
    table.add_column(table["phot_g_mean_mag"], name="mag")
    table.add_column(table["source_id"], name="objID")
    table.add_column(table["ref_epoch"], name="epoch")
    return table
