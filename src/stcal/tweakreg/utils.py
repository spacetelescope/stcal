from astropy.modeling.rotations import RotationSequence3D

__all__ = ["_wcsinfo_from_wcs_transform"]


def _wcsinfo_from_wcs_transform(wcs):
    frames = wcs.available_frames
    if "v2v3" not in frames or "world" not in frames or frames[-1] != "world":
        msg = "Unsupported WCS structure."
        raise ValueError(msg)

    # Initially get v2_ref, v3_ref, and roll_ref from
    # the v2v3 to world transform. Also get ra_ref, dec_ref
    t = wcs.get_transform(frames[-2], "world")
    for m in t:
        if isinstance(m, RotationSequence3D) and m.parameters.size == 5:
            v2_ref, nv3_ref, roll_ref, dec_ref, nra_ref = m.angles.value
            break
    else:
        msg = "Unsupported WCS structure."
        raise ValueError(msg)

    # overwrite v2_ref, v3_ref, and roll_ref with
    # values from the tangent plane when available:
    if "v2v3corr" in frames:
        # get v2_ref, v3_ref, and roll_ref from
        # the v2v3 to v2v3corr transform:
        frm1 = "v2v3vacorr" if "v2v3vacorr" in frames else "v2v3"
        tpcorr = wcs.get_transform(frm1, "v2v3corr")
        v2_ref, nv3_ref, roll_ref = tpcorr["det_to_optic_axis"].angles.value

    return {
        "v2_ref": 3600 * v2_ref,
        "v3_ref": -3600 * nv3_ref,
        "roll_ref": roll_ref,
        "ra_ref": -nra_ref,
        "dec_ref": dec_ref
    }
