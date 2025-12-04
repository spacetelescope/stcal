import concurrent.futures

import numpy as np
from astropy import units as u
from astropy import coordinates
from astropy.table import Table
from astropy.time import Time

import pyarrow.fs
import pyarrow.csv as csv
import pyarrow.parquet as pq
from cdshealpix.nested import cone_search


__all__ = ["get_catalog"]


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
    catalog = catalog[~(catalog['pmra'].mask | catalog['pmdec'].mask | catalog['parallax'].mask)]
    dt = epoch - catalog['ref_epoch']
    unitspherical = coordinates.UnitSphericalRepresentation(
        catalog['ra'] * u.deg,
        catalog['dec'] * u.deg,
    )
    xyz = unitspherical.to_cartesian().xyz
    unit_vectors = unitspherical.unit_vectors()
    rahat = unit_vectors['lon'].xyz
    dechat = unit_vectors['lat'].xyz
    earthcoord = coordinates.get_body_barycentric('earth', Time(epoch, format="decimalyear"))
    earthcoord = earthcoord.xyz.to(u.AU).value
    radpermas = np.pi / (180 * 3600 * 1000)
    newxyz = (
        xyz + rahat * dt * radpermas * catalog['pmra'] + dechat * dt * radpermas * catalog['pmdec'])
    plx = catalog['parallax']
    newxyz -= (rahat * earthcoord.dot(rahat) * plx * radpermas
               + dechat * earthcoord.dot(dechat) * plx * radpermas)
    # stars move in the opposite direction of the earth -> minus sign
    newunitspherical = coordinates.UnitSphericalRepresentation.from_cartesian(
        coordinates.CartesianRepresentation(newxyz))
    newra = newunitspherical.lon
    newdec = newunitspherical.lat
    # TODO check that units aren't lost
    catalog['ra'] = newra.to(u.deg).value
    catalog['dec'] = newdec.to(u.deg).value
    return catalog


def _get_hats_sources(gaia_dr3_uri, ra, dec, search_radius, epoch=None, columns=None):
    ra_lon = coordinates.Longitude(ra * u.deg)
    dec_lat = coordinates.Latitude(dec * u.deg)
    sr_deg = search_radius * u.deg

    # open pyarrow filesystem for accessing files
    fs, fs_path = pyarrow.fs.FileSystem.from_uri(gaia_dr3_uri)

    # get the partition_info file: Npix, Norder
    csv_file = csv.read_csv(fs.open_input_file(fs_path +  "/partition_info.csv"))
    n_order = csv_file['Norder'].to_numpy()
    n_pix = csv_file['Npix'].to_numpy()
    hats_pixels = [(int(o), int(p)) for o, p in zip(n_order, n_pix)]
    max_depth = n_order.max()
    min_depth = n_order.min()

    # perform cone search at max depth
    ipix, depth, _ = cone_search(ra_lon, dec_lat, sr_deg, max_depth)

    # generate depth 29 ranges for cone search
    dr = np.vstack((ipix << ((29 - depth) * 2), (ipix + 1) << ((29 - depth) * 2))).T

    # find matching pixels in catalog
    pixels = set()
    for (i, d) in zip(ipix, depth):
        # for each pixel in the query
        pix = (int(d), int(i))
        while pix[0] >= min_depth:
            if pix in hats_pixels:
                pixels.add(pix)
                break
            pix = (pix[0] - 1, pix[1] >> 2)
    pixels = list(pixels)

    # convert pixels to paths
    paths = []
    for pixel in pixels:
        depth, number = pixel
        directory = int(number / 10000) * 10000
        paths.append(f"{fs_path}/dataset/Norder={depth}/Dir={directory}/Npix={number}.parquet")

    # convert depth 29 ranges to pyarrow filters
    filters = []
    for hpx_range in dr:
        filters.append(
            [(SPATIAL_INDEX_COLUMN, ">=", hpx_range[0]), (SPATIAL_INDEX_COLUMN, "<", hpx_range[1])]
        )

    # read and combine data
    ds = pq.ParquetDataset(paths, filesystem=fs, filters=filters).read(columns=columns)
    table = Table({name: ds[name].to_numpy() for name in ds.column_names})

    # do the final, accurate cone search
    ra_rad = np.radians(table['ra'])
    dec_rad = np.radians(table['dec'])
    ra0 = np.radians(ra)
    dec0 = np.radians(dec)

    cos_angle = np.sin(dec_rad) * np.sin(dec0) + np.cos(dec_rad) * np.cos(dec0) * np.cos(ra_rad - ra0)

    # Clamp to valid range to avoid numerical issues
    cos_separation = np.clip(cos_angle, -1.0, 1.0)

    cos_radius = np.cos(np.radians(search_radius))
    return Table(table[cos_separation >= cos_radius])


def get_s3_catalog(
    right_ascension,
    declination,
    epoch=None,
    search_radius=0.1,
    catalog="GAIADR3_S3",
    columns=None,
    timeout=120,
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
        Timeout in seconds to wait for the catalog web service to respond. Default: 30.0 s

    Returns
    -------
    `~astropy.table.Table`
        Table of returned sources with all columns as provided by catalog
    """
    s3_url = S3_NAMES_TO_CATALOGS[catalog]

    # hats provides no way to define a timeout so use a thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_get_hats_sources, s3_url, right_ascension, declination, search_radius, epoch, list(COL_NAMES))
        table = future.result(timeout=timeout)

    if epoch:
        table = _correct_for_proper_motion(table, epoch)
    table.add_column(table["phot_g_mean_mag"], name="mag")
    table.add_column(table["source_id"], name="objID")
    table.add_column(table["ref_epoch"], name="epoch")
    return table
