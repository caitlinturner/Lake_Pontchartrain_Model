"""Utility helpers for DORADO source-based exposure workflows."""

import numpy as np
import pandas as pd
import dorado.particle_track as pt


def from_source_exposure(walk_data, elevation, regions, resolution_factor):
    """Exposure time of source-released flows within each area (chunk).

    For every chunk of the domain, this measures how long the particles in
    ``walk_data`` (all released at a single source) remain inside that chunk's
    portion of the region of interest. Unlike ``dorado.spatial.localized()``,
    particles are never re-seeded per chunk; the full source-released set is
    reused for every area.

    Parameters
    ----------
    walk_data : dict
        dorado walk-data for particles released at the source.
    elevation : ndarray
        Rasterized elevation/bathymetry grid (defines domain shape).
    regions : ndarray
        Binary region-of-interest mask, same shape as ``elevation``
        (1 = region of interest, 0 = ignore).
    resolution_factor : int
        Chunk edge length in cells. Chunk grid is
        ``rows // resolution_factor`` by ``cols // resolution_factor``.

    Returns
    -------
    list
        Row-major list of per-chunk exposure-time data. Entries for chunks
        that do not overlap the region of interest are ``np.nan``.
    """
    rows, cols = elevation.shape
    n_chunks_rows = rows // resolution_factor
    n_chunks_cols = cols // resolution_factor
    roi = regions == 1

    exposure_data = []
    for chunk_row in range(n_chunks_rows):
        for chunk_col in range(n_chunks_cols):
            start_row = chunk_row * resolution_factor
            end_row = min((chunk_row + 1) * resolution_factor, rows)
            start_col = chunk_col * resolution_factor
            end_col = min((chunk_col + 1) * resolution_factor, cols)

            if np.any(roi[start_row:end_row, start_col:end_col]):
                regions_chunk = np.zeros((rows, cols), dtype="int")
                regions_chunk[start_row:end_row, start_col:end_col] = (
                    roi[start_row:end_row, start_col:end_col].astype(int)
                )
                exposure_data.append(
                    pt.exposure_time(walk_data, regions_chunk, verbose=False)
                )
            else:
                exposure_data.append(np.nan)

    return exposure_data


def build_daily_series(schedule):
    """Return a daily pandas Series (index=date, value=particles that day)."""
    stype = schedule["type"]

    if stype == "particles_csv":
        df = pd.read_csv(schedule["path"])
        d = pd.to_datetime(df[schedule.get("date_col", "datetime")]).dt.floor("D")
        v = pd.to_numeric(
            df[schedule.get("value_col", "particle_number")], errors="coerce"
        ).fillna(0.0)
        s = pd.Series(v.values, index=d).groupby(level=0).sum().sort_index()

    elif stype == "discharge_csv":
        df = pd.read_csv(schedule["path"])
        d = pd.to_datetime(df[schedule.get("date_col", "date")]).dt.floor("D")
        q = pd.to_numeric(df[schedule.get("q_col", "discharge")], errors="coerce").fillna(
            0.0
        )
        s = pd.Series(q.values, index=d).groupby(level=0).sum().sort_index()
        if schedule.get("positive_only", True):
            s = s[s > 0]
        smooth = int(schedule.get("smooth_days", 1))
        if smooth > 1:
            s = s.rolling(window=smooth, center=True, min_periods=1).mean()
        s = s * float(schedule.get("scale", 1.0))

    elif stype == "constant":
        start = pd.Timestamp(schedule["start"]).floor("D")
        end = pd.Timestamp(schedule["end"]).floor("D")
        idx = pd.date_range(start, end, freq="D")
        s = pd.Series(float(schedule["rate_per_day"]), index=idx)

    elif stype == "series":
        d = pd.to_datetime(pd.Index(schedule["dates"])).floor("D")
        s = pd.Series(np.asarray(schedule["counts"], dtype=float), index=d).sort_index()

    else:
        raise ValueError(f"Unknown schedule type: {stype}")

    s = s.clip(lower=0).astype(float)
    s.index.name = "date"
    return s


def resolve_seed_cells(location, elevation, resolution, origin_xy, utm_proj=None):
    """Return (seed_xloc, seed_yloc) index lists (rows, cols) for one source."""
    ltype = location["type"]
    rows, cols = elevation.shape

    if ltype == "points":
        seed_x = [int(r) for r, c in location["cells"]]
        seed_y = [int(c) for r, c in location["cells"]]

    elif ltype == "point":
        n = int(location.get("n_cells", 1))
        seed_x = [int(location["row"])] * n
        seed_y = [int(location["col"])] * n

    elif ltype == "line":
        if "row" in location and "cols" in location:
            c0, c1 = location["cols"]
            seed_y = list(range(int(c0), int(c1) + 1))
            seed_x = [int(location["row"])] * len(seed_y)
        elif "col" in location and "rows" in location:
            r0, r1 = location["rows"]
            seed_x = list(range(int(r0), int(r1) + 1))
            seed_y = [int(location["col"])] * len(seed_x)
        else:
            raise ValueError("line needs ('row' & 'cols') or ('col' & 'rows').")

    elif ltype == "mask":
        mask = (
            location["mask"]
            if "mask" in location
            else (location["grid"] == location["equals"])
        )
        idx = np.argwhere(np.asarray(mask))
        seed_x = idx[:, 0].tolist()
        seed_y = idx[:, 1].tolist()

    elif ltype == "latlon":
        if utm_proj is None:
            raise ValueError("latlon location requires utm_proj.")
        lons = np.atleast_1d(location["lon"]).astype(float)
        lats = np.atleast_1d(location["lat"]).astype(float)
        xs, ys = utm_proj(lons, lats)
        coords = [(round(float(px), 2), round(float(py), 2)) for px, py in zip(xs, ys)]
        inds = pt.coord2ind(coords, origin_xy, elevation.shape, resolution)
        seed_x = [int(r) for r, c in inds]
        seed_y = [int(c) for r, c in inds]

    else:
        raise ValueError(f"Unknown location type: {ltype}")

    # Keep only in-bounds, wet (non-NaN elevation) cells
    sx, sy = [], []
    for r, c in zip(seed_x, seed_y):
        if 0 <= r < rows and 0 <= c < cols and not np.isnan(elevation[r, c]):
            sx.append(int(r))
            sy.append(int(c))
    if not sx:
        raise ValueError(f"No valid wet, in-bounds seed cells for location {location}.")
    return sx, sy


def daily_to_steps(daily_counts, steps_per_day):
    """Spread each day's particle total across the steps within that day."""
    out = []
    for n_particles in np.round(np.asarray(daily_counts)).astype(int):
        base, rem = divmod(int(n_particles), steps_per_day)
        arr = np.full(steps_per_day, base, dtype=int)
        if rem > 0:
            arr[np.linspace(0, steps_per_day - 1, rem, dtype=int)] += 1
        out.append(arr)
    return np.concatenate(out) if out else np.array([], dtype=int)