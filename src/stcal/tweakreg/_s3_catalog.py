import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import hats

# No support for GAIADR2 or GAIADR1
# GAIAREFCAT same as GAIADR3?
SUPPORTED = ["GAIAREFCAT", "GAIADR3"]
S3_URL = "s3://stpubdata/gaia/gaia_dr3/public/hats/gaia/"
MAX_PYARROW_FILTERS = 10
SPATIAL_INDEX_COLUMN = "_healpix_29"
TIMEOUT = 30.0  # in seconds FIXME unused
# source_id -> objID
# epoch?
# mag? phot_g_mean_mag?
# pm?
COL_NAMES = (
    "ra",
    "ra_error",
    "dec",
    "dec_error",
    "parallax",
    "parallax_error",
    "source_id",
    "ref_epoch",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "phot_g_mean_mag",
)

def _generate_pyarrow_filters_from_moc(filtered_catalog):
    pyarrow_filter = []
    if SPATIAL_INDEX_COLUMN not in filtered_catalog.schema.names:
        return pyarrow_filter
    if filtered_catalog.moc is None:
       return pyarrow_filter
    depth_array = filtered_catalog.moc.to_depth29_ranges
    if len(depth_array) > MAX_PYARROW_FILTERS:
        starts = depth_array.T[0]
        ends = depth_array.T[1]
        diffs = starts[1:] - ends[:-1]
        max_diff_inds = np.argpartition(diffs, -MAX_PYARROW_FILTERS)[-MAX_PYARROW_FILTERS:]
        max_diff_inds = np.sort(max_diff_inds)
        reduced_filters = []
        for i_start, i_end in zip(np.concat(([0], max_diff_inds)), np.concat((max_diff_inds, [-1]))):
            reduced_filters.append([starts[i_start], ends[i_end]])
        depth_array = np.array(reduced_filters)
    for hpx_range in depth_array:
        pyarrow_filter.append(
            [(SPATIAL_INDEX_COLUMN, ">=", hpx_range[0]), (SPATIAL_INDEX_COLUMN, "<", hpx_range[1])]
        )
    return pyarrow_filter


def filter_hc_catalog(cat, ra, dec, radius_arcsec, columns=None):
    # filter by spatial area
    filtered_cat = cat.filter_by_cone(ra, dec, radius_arcsec)
    paths = [
        hats.io.paths.pixel_catalog_file(
            filtered_cat.catalog_base_dir, p, npix_suffix=filtered_cat.catalog_info.npix_suffix
        ) for p in filtered_cat.get_healpix_pixels()
    ]
    pyarrow_filter = _generate_pyarrow_filters_from_moc(filtered_cat)
    list_dfs = [
        pq.ParquetDataset(str(path), filters=pyarrow_filter).read(columns=columns).to_pandas()
        for path in paths
    ]
    df = pd.concat(list_dfs)
    return hats.search.region_search.cone_filter(df, ra, dec, radius_arcsec, filtered_cat.catalog_info)


def get_gaia_DR3_sources(gaia_dr3_uri, ra, dec, radius_arcsec, columns=None):
    cat = hats.read_hats(gaia_dr3_uri)
    return filter_hc_catalog(cat, ra, dec, radius_arcsec, columns=columns)


def get_catalog(
    right_ascension,
    declination,
    epoch=2016.0,
    search_radius=0.1,
    catalog="GAIADR3",
    timeout=TIMEOUT,
    columns=None,
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
        (in decimal year). Default: 2016.0

    search_radius : float, optional
        Search radius (in decimal degrees) from field-of-view center to use
        for sources from catalog.  Default: 0.1 degrees

    catalog : str, optional
        Name of catalog to query, as defined by web-service. Default: 'GAIADR3'

    timeout : float, optional
        Timeout in seconds to wait for the catalog web service to respond. Default: 30.0 s

    Returns
    -------
    `~astropy.table.Table`
        Table of returned sources with all columns as provided by catalog
    """
    # TODO epoch?
    # TODO columns?
    if columns is None:
        columns = COL_NAMES
    # TODO timeout?
    radius_arcsec = search_radius * 3600
    # TODO convert pandas to astropy table
    # TODO map columns back to expected names (or update downstream code)
    # TODO correct for proper motion
    return get_gaia_DR3_sources(S3_URL, right_ascension, declination, radius_arcsec, list(columns))

