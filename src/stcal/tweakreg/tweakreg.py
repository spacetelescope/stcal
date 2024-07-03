from __future__ import annotations

import math
import warnings
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from tweakwcs.correctors import JWSTWCSCorrector
from tweakwcs.imalign import align_wcs
from tweakwcs.matchutils import XYXYMatch

from stcal.alignment import SupportsDataWithWcs, wcs_from_footprints

from .astrometric_utils import create_astrometric_catalog


def _oxford_or_str_join(str_list):
    nelem = len(str_list)
    if not nelem:
        return "N/A"
    str_list = list(map(repr, str_list))
    if nelem == 1:
        return str_list
    if nelem == 2:
        return str_list[0] + " or " + str_list[1]
    return ", ".join(map(repr, str_list[:-1])) + ", or " + repr(str_list[-1])


_SQRT2 = math.sqrt(2.0)
SINGLE_GROUP_REFCAT = ["GAIADR3", "GAIADR2", "GAIADR1"]
_SINGLE_GROUP_REFCAT_STR = _oxford_or_str_join(SINGLE_GROUP_REFCAT)


class TweakregError(BaseException):
    pass


def relative_align(correctors: list[JWSTWCSCorrector],
                   searchrad: float = 2.0,
                   separation: float = 1.0,
                   use2dhist: bool = True,
                   tolerance: float = 0.7,
                   xoffset: float = 0.0,
                   yoffset: float = 0.0,
                   enforce_user_order: bool = False,
                   expand_refcat: bool = False,
                   minobj: int = 15,
                   fitgeometry: str = "rshift",
                   nclip: int = 3,
                   sigma: float = 3.0,
                   align_to_abs_refcat: bool = False,
                   ) -> tuple[list[JWSTWCSCorrector], bool]:

    if separation <= _SQRT2 * tolerance:
        msg = "Parameter 'separation' must be larger than 'tolerance' by at \
            least a factor of sqrt(2) to avoid source confusion."
        raise TweakregError(msg)

    # align images:
    xyxymatch = XYXYMatch(
        searchrad=searchrad,
        separation=separation,
        use2dhist=use2dhist,
        tolerance=tolerance,
        xoffset=xoffset,
        yoffset=yoffset
    )

    try:
        align_wcs(
            correctors,
            refcat=None,
            enforce_user_order=enforce_user_order,
            expand_refcat=expand_refcat,
            minobj=minobj,
            match=xyxymatch,
            fitgeom=fitgeometry,
            nclip=nclip,
            sigma=(sigma, "rmse")
        )
        local_align_failed = False

    except ValueError as e:
        msg = e.args[0]
        if (msg == "Too few input images (or groups of images) with "
                "non-empty catalogs."):
            local_align_failed = True
            if not align_to_abs_refcat:
                msg += "At least two exposures are required for image alignment. Nothing to do."
                raise TweakregError(msg) from None
        else:
            raise

    except RuntimeError as e:
        msg = e.args[0]
        if msg.startswith("Number of output coordinates exceeded allocation"):
            # we need at least two exposures to perform image alignment
            msg += "Multiple sources within specified tolerance \
                    matched to a single reference source. Try to \
                    adjust 'tolerance' and/or 'separation' parameters."
            raise TweakregError(msg) from None
        raise

    with warnings.catch_warnings(record=True) as w:
        is_small = _is_wcs_correction_small(correctors,
                                            use2dhist,
                                            searchrad,
                                            tolerance,
                                            xoffset,
                                            yoffset)
        warning_msg = "".join([str(mess.message) for mess in w])
        if not local_align_failed and not is_small:
            if align_to_abs_refcat:
                warning_msg += " Skipping relative alignment (stage 1)..."
                warnings.warn(warning_msg, stacklevel=1)
            else:
                raise TweakregError(warning_msg)

    return correctors, local_align_failed


def absolute_align(correctors: list[JWSTWCSCorrector],
                   abs_refcat: str,
                   ref_image: SupportsDataWithWcs,
                   save_abs_catalog: bool = False,
                   abs_catalog_output_dir: str | None = None,
                   abs_searchrad: float = 6.0,
                   abs_separation: float = 1.0,
                   abs_use2dhist: bool = True,
                   abs_tolerance: float = 0.7,
                   abs_minobj: int = 15,
                   abs_fitgeometry: str = "rshift",
                   abs_nclip: int = 3,
                   abs_sigma: float = 3.0,
                   local_align_failed: bool = False,) -> list[JWSTWCSCorrector]:

    if abs_separation <= _SQRT2 * abs_tolerance:
        msg = "Parameter 'abs_separation' must be larger than 'abs_tolerance' by at \
            least a factor of sqrt(2) to avoid source confusion."
        raise TweakregError(msg)

    ref_cat = _parse_refcat(abs_refcat,
                            ref_image,
                            correctors,
                            save_abs_catalog=save_abs_catalog,
                            output_dir=abs_catalog_output_dir)

    # Check that there are enough GAIA sources for a reliable/valid fit
    num_ref = len(ref_cat)
    if num_ref < abs_minobj:
        msg = f"Not enough sources ({num_ref}) in the reference catalog \
            for the single-group alignment step to perform a fit. \
            Skipping alignment to the input reference catalog!"
        raise TweakregError(msg)

    # align images:
    # Update to separation needed to prevent confusion of sources
    # from overlapping images where centering is not consistent or
    # for the possibility that errors still exist in relative overlap.
    xyxymatch_gaia = XYXYMatch(
        searchrad=abs_searchrad,
        separation=abs_separation,
        use2dhist=abs_use2dhist,
        tolerance=abs_tolerance,
        xoffset=0.0,
        yoffset=0.0
    )

    # Set group_id to same value so all get fit as one observation
    # The assigned value, 987654, has been hard-coded to make it
    # easy to recognize when alignment to GAIA was being performed
    # as opposed to the group_id values used for relative alignment
    # earlier in this step.
    for corrector in correctors:
        corrector.meta["group_id"] = 987654
        if ("fit_info" in corrector.meta and
                "REFERENCE" in corrector.meta["fit_info"]["status"]):
            del corrector.meta["fit_info"]

    # Perform fit
    try:
        align_wcs(
            correctors,
            refcat=ref_cat,
            enforce_user_order=True,
            expand_refcat=False,
            minobj=abs_minobj,
            match=xyxymatch_gaia,
            fitgeom=abs_fitgeometry,
            nclip=abs_nclip,
            sigma=(abs_sigma, "rmse")
        )
    except ValueError as e:
        msg = e.args[0]
        if (msg == "Too few input images (or groups of images) with "
                "non-empty catalogs."):
            msg += "At least one exposure is required to align images \
                    to an absolute reference catalog. Alignment to an \
                    absolute reference catalog will not be performed."
            if local_align_failed:
                msg += " Nothing to do. Skipping 'TweakRegStep'..."
                raise TweakregError(msg) from None
            warnings.warn(msg, stacklevel=1)
        else:
            raise

    except RuntimeError as e:
        msg = e.args[0]
        if msg.startswith("Number of output coordinates exceeded allocation"):
            # we need at least two exposures to perform image alignment
            msg += "Multiple sources within specified tolerance \
                    matched to a single reference source. Try to \
                    adjust 'tolerance' and/or 'separation' parameters. \
                    Alignment to an absolute reference catalog will \
                    not be performed."
            if local_align_failed:
                msg += "Skipping 'TweakRegStep'..."
                raise TweakregError(msg) from None
            warnings.warn(msg, stacklevel=1)
        else:
            raise

    return correctors


def _parse_refcat(abs_refcat: str,
                  ref_model: SupportsDataWithWcs,
                  correctors: list,
                  save_abs_catalog: bool = False,
                  output_dir: str | None = None) -> Table:
    """
    Figure out if abs_refcat is an input filename or
    the name of a GAIA catalog. If the former, load it,
    and if the latter, retrieve that catalog from the Web.
    If desired, save the reference catalog in the specified directory.
    """
    if save_abs_catalog:
        root = f"fit_{abs_refcat.lower()}_ref.ecsv"
        output_name = Path(root) if output_dir is None \
            else Path(output_dir) / root
    else:
        output_name = None

    abs_refcat = abs_refcat.strip()
    gaia_cat_name = abs_refcat.upper()
    if gaia_cat_name in SINGLE_GROUP_REFCAT:

        epoch = Time(ref_model.meta.observation.date).decimalyear

        # combine all aligned wcs to compute a new footprint to
        # filter the absolute catalog sources
        combined_wcs = wcs_from_footprints(
            None,
            refmodel=ref_model,
            wcs_list=[corrector.wcs for corrector in correctors],
        )

        return create_astrometric_catalog(
            combined_wcs, epoch,
            catalog=gaia_cat_name,
            output=output_name,
        )

    if Path.is_file(Path(abs_refcat)):
        return Table.read(abs_refcat)

    msg = f"Invalid 'abs_refcat' value: {abs_refcat}. 'abs_refcat' must be \
            a path to an existing file name or one of the supported \
            reference catalogs: {_SINGLE_GROUP_REFCAT_STR}."
    raise ValueError(msg)


def _is_wcs_correction_small(correctors,
                             use2dhist=True,
                             searchrad=2.0,
                             tolerance=0.7,
                             xoffset=0.0,
                             yoffset=0.0):
    # check for a small wcs correction, it should be small
    if use2dhist:
        max_corr = 2 * (searchrad + tolerance) * u.arcsec
    else:
        max_corr = 2 * (max(abs(xoffset), abs(yoffset)) +
                        tolerance) * u.arcsec
    for corrector in correctors:
        aligned_skycoord = _wcs_to_skycoord(corrector.wcs)
        original_skycoord = corrector.meta["original_skycoord"]
        separation = original_skycoord.separation(aligned_skycoord)
        if not (separation < max_corr).all():
            # Large corrections are typically a result of source
            # mis-matching or poorly-conditioned fit. Skip such models.
            msg = f"WCS has been tweaked by more than {10 * tolerance} arcsec"
            warnings.warn(msg, stacklevel=1)
            return False
    return True


def _wcs_to_skycoord(wcs):
    ra, dec = wcs.footprint(axis_type="spatial").T
    return SkyCoord(ra=ra, dec=dec, unit="deg")


def _filter_catalog_by_bounding_box(catalog: Table, bounding_box: list[float]) -> Table:
    """
    Given a catalog of x,y positions, only return sources that fall
    inside the bounding box.
    """
    if bounding_box is None:
        return catalog

    # filter out sources outside the WCS bounding box
    ((xmin, xmax), (ymin, ymax)) = bounding_box
    x = catalog["x"]
    y = catalog["y"]
    mask = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
    return catalog[mask]


def construct_wcs_corrector(image_model: SupportsDataWithWcs,
                            catalog: Table) -> JWSTWCSCorrector:
    """
    pre-compute skycoord here so we can later use it
    to check for a small wcs correction.
    """
    catalog = _filter_catalog_by_bounding_box(
        catalog,
        image_model.meta.wcs.bounding_box
    )

    wcs = image_model.meta.wcs
    refang = image_model.meta.wcsinfo.instance
    return JWSTWCSCorrector(
        wcs=image_model.meta.wcs,
        wcsinfo={"roll_ref": refang["roll_ref"],
                 "v2_ref": refang["v2_ref"],
                 "v3_ref": refang["v3_ref"]},
        # catalog and group_id are required meta
        meta={
            "catalog": catalog,
            "name": catalog.meta.get("name"),
            "group_id": image_model.meta.group_id,
            "original_skycoord": _wcs_to_skycoord(wcs),
        }
    )
