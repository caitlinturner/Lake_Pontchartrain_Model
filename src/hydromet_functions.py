"""Helper functions for creating model dataset.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataretrieval import nwis
from noaa_coops import Station
from pvlib.location import Location
from scipy.signal import butter, filtfilt


# Runtime context values are injected from notebook cells.
WINDOW_MMDD_START = "01-01"
WINDOW_MMDD_END = "12-31"
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"

USGS_SALINITY_PREFER_CONDUCTANCE: set[str] = set()
USGS_WATER_TEMP_SITE = ""
WT_BACKUP_STATIONS: list[str] = []
USGS_SALINITY_SITE = ""
USGS_SALINITY_BACKUPS: list[str] = []
WL_STATION = ""
WL_BACKUP_STATIONS: list[str] = []
WL_GAP_THRESHOLD_H = 24
NOAA_DATUM = "MLLW"
MET_STATION = ""
MET_SMOOTHING_HOURS = 6
INCLUDE_TEMPERATURE = True
INCLUDE_SALINITY = True

RIVERS_INFO: list[dict] = []
N_PTS_RIVER = 2
SPILLWAY_BND: dict = {}
OPEN_BND: dict = {}
DFM_OUT_ROOT = Path("outputs/dfm_forcings")
DFM_REF_DATE = pd.Timestamp("2001-01-01")
DFM_FILES: dict = {}
RIVER_SALINITY_PPT = 0.2
RIVER_TEMP_SMOOTH_HOURS = 24 * 21  # 2 Weeks
OCEAN_TEMP_SMOOTH_HOURS = 24 * 7   # 1 Week

spillway_by_year: dict[int, pd.DataFrame] = {}
trib_by_year: dict[int, pd.DataFrame] = {}
wl_by_year: dict[int, pd.DataFrame] = {}
wind_by_year: dict[int, pd.DataFrame] = {}
pres_by_year: dict[int, pd.DataFrame] = {}
airT_by_year: dict[int, pd.DataFrame] = {}
watT_by_year: dict[int, pd.DataFrame] = {}
sal_by_year: dict[int, pd.DataFrame] = {}
openmeteo_by_year: dict[int, pd.DataFrame] = {}
solar_by_year: dict[int, pd.DataFrame] = {}
rivT_by_year: dict[int, pd.DataFrame] = {}
TRIB_DV_CACHE: dict[tuple[str, int], pd.DataFrame | None] = {}

_FIELD_W = 20


def configure_hydromet_context(**kwargs):
    """Inject notebook runtime constants/datasets into this module namespace."""
    globals().update(kwargs)


def year_window(year):
    return (pd.Timestamp(f"{year}-{WINDOW_MMDD_START}"), pd.Timestamp(f"{year}-{WINDOW_MMDD_END}"))


def hourly_index_for_year(year):
    start, end = year_window(year)
    # Add one day to end to include the full end date, then use inclusive="left" to exclude the day after
    return pd.date_range(start, end + pd.Timedelta(days=1), freq="h", inclusive="left")


def _ensure_naive_datetime(df, col="datetime"):
    s = pd.to_datetime(df[col], errors="coerce")
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            pass
    df[col] = s
    return df


def to_hourly_spline(df, value_col, hourly_index, roll_win=3):
    s = df.set_index("datetime")[value_col] if "datetime" in df.columns else df[value_col]
    s.index = pd.to_datetime(s.index, errors="coerce")
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = s.resample("h").mean().reindex(hourly_index)
    s = s.rolling(roll_win, center=True, min_periods=1).mean()
    # Use linear interpolation instead of cubic spline to avoid wild oscillations
    # (especially important for daily discharge data)
    try:
        s = s.interpolate(method="linear", limit_direction="both")
    except Exception:
        s = s.interpolate(method="time", limit_direction="both")
    return s.to_frame(name=value_col).reset_index().rename(columns={"index": "datetime"})


def hourly_ma_spline(df, value_col, win=3):
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    s = df[value_col].resample("h").mean().rolling(win, center=True, min_periods=1).mean()
    try:
        s = s.interpolate(method="spline", order=3, limit_direction="both")
    except Exception:
        s = s.interpolate(method="time", limit_direction="both")
    return s.to_frame().rename_axis("datetime").reset_index()


def circular_mean_deg(series_deg):
    rad = np.deg2rad(pd.to_numeric(series_deg, errors="coerce").dropna().values)
    if rad.size == 0:
        return np.nan
    mean = np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
    return (mean + 360.0) % 360.0


def wind_to_hourly_uv(wind_df, hourly_index, roll_win=3):
    w = wind_df.copy()
    w = w.set_index(pd.to_datetime(w["datetime"], errors="coerce")).drop(columns=["datetime"])
    if getattr(w.index, "tz", None) is not None:
        w.index = w.index.tz_convert("UTC").tz_localize(None)

    rad = np.deg2rad(pd.to_numeric(w["direction"], errors="coerce"))
    spd = pd.to_numeric(w["speed"], errors="coerce")
    u = spd * np.sin(rad)
    v = spd * np.cos(rad)

    uv = pd.DataFrame({"u": u, "v": v}).resample("h").mean().reindex(hourly_index)
    uv = uv.rolling(roll_win, center=True, min_periods=1).mean()
    uv = uv.interpolate(method="time", limit_direction="both")

    new_spd = np.sqrt(uv["u"] ** 2 + uv["v"] ** 2)
    new_dir = (np.degrees(np.arctan2(uv["u"], uv["v"])) + 360.0) % 360.0

    out = pd.DataFrame({"speed": new_spd.values, "direction": new_dir.values}, index=hourly_index).rename_axis("datetime").reset_index()
    return out


def smooth_hourly_wind(wind_df, hourly_index, win=6):
    w = wind_df.copy()
    if w.empty or "datetime" not in w.columns:
        return pd.DataFrame({"datetime": hourly_index, "speed": np.nan, "direction": np.nan})

    w = w.set_index(pd.to_datetime(w["datetime"], errors="coerce")).drop(columns=["datetime"])
    if getattr(w.index, "tz", None) is not None:
        w.index = w.index.tz_convert("UTC").tz_localize(None)

    rad = np.deg2rad(pd.to_numeric(w["direction"], errors="coerce"))
    spd = pd.to_numeric(w["speed"], errors="coerce")
    u = spd * np.sin(rad)
    v = spd * np.cos(rad)

    uv = pd.DataFrame({"u": u, "v": v}).resample("h").mean().reindex(hourly_index)
    uv = uv.rolling(win, center=True, min_periods=1).mean()
    uv = uv.interpolate(method="time", limit_direction="both")

    speed = np.sqrt(uv["u"] ** 2 + uv["v"] ** 2)
    direction = (np.degrees(np.arctan2(uv["u"], uv["v"])) + 360.0) % 360.0

    return pd.DataFrame(
        {"datetime": hourly_index, "speed": speed.values, "direction": direction.values}
    )


def _safe_lowpass(series, cutoff_hours=6.0, order=6):
    s = pd.to_numeric(series, errors="coerce").interpolate(method="time", limit_direction="both")
    try:
        dt_seconds = series.index.to_series().diff().dt.total_seconds().dropna().median()
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            dt_seconds = 360.0
    except Exception:
        dt_seconds = 360.0
    wn = max(min(2.0 * dt_seconds / (cutoff_hours * 3600.0), 0.99), 1e-6)
    b, a = butter(order, wn, btype="low")
    edge = 3 * (max(len(a), len(b)) - 1)
    if len(s) <= edge + 1:
        win = max(int(round((cutoff_hours * 3600.0) / dt_seconds)), 3)
        return s.rolling(win, center=True, min_periods=1).mean()
    return pd.Series(filtfilt(b, a, s.values, method="pad"), index=s.index)


def _find_year_columns(df, year, date_col):
    tok = str(year)
    norm = {c: str(c).strip().lower() for c in df.columns}
    return [c for c in df.columns if c != date_col and tok in norm[c]]


def load_usace_bcs_waterlevel_txt(
    txt_path,
    date_col="date",
    waterlevel_col="waterlevel",
    missing_marker="M",
    feet_to_m=0.3048,
    datum_offset_m=0.27,
):
    """Load Bonnet Carre USACE stage text and parse USACE missing markers.

    Parameters
    ----------
    txt_path : str or Path
        Path to USACE text file, e.g. data/external/waterlevel_BCS_USACE.txt.
    date_col : str
        Date-time column name.
    waterlevel_col : str
        Stage column name in feet.
    missing_marker : str
        Marker used by USACE for missing values (typically "M").
    feet_to_m : float
        Feet-to-meter conversion.
    datum_offset_m : float
        Datum adjustment in meters applied after conversion.

    Returns
    -------
    pd.DataFrame
        Columns: datetime, waterlevel_ft, waterlevel_m, waterlevel_missing.
    """
    df = pd.read_csv(txt_path, sep="\t", header=0)
    if date_col not in df.columns or waterlevel_col not in df.columns:
        raise ValueError(
            f"Expected columns '{date_col}' and '{waterlevel_col}' in {txt_path}."
        )

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(df[date_col], errors="coerce")

    raw = df[waterlevel_col]
    raw_str = raw.astype(str).str.strip()
    out["waterlevel_missing"] = raw_str.str.upper().eq(str(missing_marker).upper())

    out["waterlevel_ft"] = pd.to_numeric(raw, errors="coerce")
    out["waterlevel_m"] = out["waterlevel_ft"] * feet_to_m - datum_offset_m

    out = out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return out


def load_usace_bonnet_carre_openings(
    xlsx_path,
    sheet=0,
    date_col="date",
    year_col_prefix="year",
    missing_marker="M",
):
    """Read all Bonnet Carre opening columns and convert them to a daily year/month/day table.

    The workbook stores openings in separate columns (e.g., year2019a, year2019b).
    This function sums same-day flows across columns for the same year and tracks
    whether any source value was marked missing.

    Returns
    -------
    pd.DataFrame
        Columns: year, month, day, opening_cms, opening_missing.
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet)
    if date_col not in raw.columns:
        raise ValueError(f"Missing '{date_col}' column in opening workbook: {xlsx_path}")

    d = pd.to_datetime(raw[date_col], errors="coerce")
    base = pd.DataFrame({"month": d.dt.month, "day": d.dt.day})

    ycols = [
        c
        for c in raw.columns
        if str(c).strip().lower().startswith(str(year_col_prefix).lower())
    ]
    if not ycols:
        return pd.DataFrame(
            columns=["year", "month", "day", "opening_cms", "opening_missing"]
        )

    frames = []
    miss_tok = str(missing_marker).upper()
    for col in ycols:
        m = re.search(r"(\d{4})", str(col))
        if not m:
            continue
        yr = int(m.group(1))

        s_raw = raw[col]
        s_num = pd.to_numeric(s_raw, errors="coerce")
        s_miss = s_raw.astype(str).str.strip().str.upper().eq(miss_tok)

        tmp = base.copy()
        tmp["year"] = yr
        tmp["opening_cms"] = s_num
        tmp["opening_missing"] = s_miss
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(
            columns=["year", "month", "day", "opening_cms", "opening_missing"]
        )

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["month", "day"])

    out = (
        long_df.groupby(["year", "month", "day"], as_index=False)
        .agg(
            opening_cms=("opening_cms", lambda s: s.sum(min_count=1)),
            opening_missing=("opening_missing", "any"),
        )
        .sort_values(["year", "month", "day"])
        .reset_index(drop=True)
    )
    return out


def compute_bcs_leakage_with_opening_mask(
    waterlevel_txt_path,
    openings_xlsx_path,
    openings_sheet=0,
    openings_date_col="date",
    opening_threshold_cms=0.0,
    missing_opening_policy="nan",
    crest_elevation_m=(5.18 + 4.57) / 2,
    spillway_length_m=2135.0,
    gravity_mps2=9.81,
    slat_spacing_m=0.075,
    bays=350,
    timbers=20,
):
    """Compute Bonnet Carre leakage and suppress leakage during spillway openings.

    Rules applied:
    - Leakage is calculated from USACE stage data.
    - If opening discharge > opening_threshold_cms, leakage is forced to 0.
    - Missing water level marker ("M") yields NaN leakage for that timestamp.
    - Missing opening entries are handled via `missing_opening_policy`.

    Parameters
    ----------
    missing_opening_policy : {"nan", "zero", "leakage"}
        - "nan": set leakage to NaN when opening data is missing.
        - "zero": set leakage to 0 when opening data is missing.
        - "leakage": keep calculated leakage when opening data is missing.

    Returns
    -------
    pd.DataFrame
        Includes datetime, water level, opening fields, raw leakage, and final leakage.
    """
    wl = load_usace_bcs_waterlevel_txt(waterlevel_txt_path)
    op = load_usace_bonnet_carre_openings(
        openings_xlsx_path,
        sheet=openings_sheet,
        date_col=openings_date_col,
    )

    out = wl.copy()
    out["year"] = out["datetime"].dt.year
    out["month"] = out["datetime"].dt.month
    out["day"] = out["datetime"].dt.day

    if op.empty:
        out["opening_cms"] = 0.0
        out["opening_missing"] = False
    else:
        out = out.merge(op, on=["year", "month", "day"], how="left")
        out["opening_cms"] = pd.to_numeric(out["opening_cms"], errors="coerce").fillna(0.0)
        out["opening_missing"] = out["opening_missing"].fillna(False).astype(bool)

    out["opening_active"] = out["opening_cms"] > float(opening_threshold_cms)

    # Leakage equation from McCorquodale (2009) notebook implementation.
    out["H_m"] = (out["waterlevel_m"] - float(crest_elevation_m)).clip(lower=0.0)
    out["Cd"] = 0.385 + 0.0025 * (out["H_m"] / float(crest_elevation_m))

    width_m = float(bays) * (float(timbers) + 1.0) * float(slat_spacing_m)
    fraction = width_m / float(spillway_length_m)
    out["leakage_raw_cms"] = (
        fraction
        * float(spillway_length_m)
        * out["Cd"]
        * np.sqrt(2.0 * float(gravity_mps2))
        * np.power(out["H_m"], 1.5)
    )

    out["leakage_cms"] = out["leakage_raw_cms"]
    out.loc[out["opening_active"], "leakage_cms"] = 0.0

    policy = str(missing_opening_policy).strip().lower()
    if policy == "nan":
        out.loc[out["opening_missing"], "leakage_cms"] = np.nan
    elif policy == "zero":
        out.loc[out["opening_missing"], "leakage_cms"] = 0.0
    elif policy != "leakage":
        raise ValueError(
            "missing_opening_policy must be one of: 'nan', 'zero', 'leakage'."
        )

    return out.sort_values("datetime").reset_index(drop=True)


def load_spillway(xlsx, sheet, date_col, year):
    raw = pd.read_excel(xlsx, sheet_name=sheet)
    if date_col not in raw.columns:
        raise ValueError(f"Missing '{date_col}' column in spillway Excel.")
    d = pd.to_datetime(raw[date_col], errors="coerce")
    d_fixed = pd.to_datetime(dict(year=np.full(len(d), year, dtype=int), month=d.dt.month, day=d.dt.day), errors="coerce")
    ycols = _find_year_columns(raw, year, date_col)
    if not ycols:
        print(f"  [spillway] no {year} column in USACE spreadsheet - assuming BCS was closed all year (Q=0)")
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        return pd.DataFrame({"datetime": days, "Q_cms": np.zeros(len(days))})
    q_sum = raw[ycols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    out = pd.DataFrame({"datetime": d_fixed, "Q_cms": q_sum}).dropna(subset=["datetime"])
    s, e = year_window(year)
    return out.loc[(out["datetime"] >= s) & (out["datetime"] <= e)].sort_values("datetime").reset_index(drop=True)


def load_tributary_scaled(site_no, year, conv, amp):
    s, e = year_window(year)
    empty = pd.DataFrame(columns=["datetime", "Q_cms"])
    cache_key = (str(site_no), int(year))
    dv = TRIB_DV_CACHE.get(cache_key, None)
    if cache_key not in TRIB_DV_CACHE:
        last_err = None
        for attempt in range(1, 4):
            try:
                dv, _ = nwis.get_dv(sites=site_no, parameterCd="00060", start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"))
                break
            except Exception as err:
                last_err = err
                if attempt < 3:
                    print(f"  [warn] site {site_no} {year}: USGS fetch attempt {attempt}/3 failed ({type(err).__name__}); retrying")
                else:
                    print(f"  [warn] site {site_no} {year}: USGS fetch failed ({type(err).__name__}: {err})")
                    dv = None
        TRIB_DV_CACHE[cache_key] = dv
        if dv is None:
            return empty
    elif dv is None:
        # Cache records a failed fetch for this site/year; keep rivers consistent.
        return empty

    candidates = [c for c in dv.columns if "00060" in c]
    if not candidates:
        print(f"  [warn] site {site_no} {year}: no 00060 column in response (cols={list(dv.columns)})")
        return empty
    col = "00060_Mean" if "00060_Mean" in dv.columns else candidates[0]
    df = dv[[col]].reset_index()
    df.columns = ["datetime", col]
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["Q_cms"] = pd.to_numeric(df[col], errors="coerce") * conv * amp
    return df[["datetime", "Q_cms"]].dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


def coops_water_level_native(start_str, end_str, station_id, datum="MLLW"):
    try:
        stn = Station(id=station_id)
        raw = stn.get_data(begin_date=start_str, end_date=end_str, product="water_level", datum=datum, units="metric", time_zone="gmt")
        df = pd.DataFrame({"wl": raw.v})
        df.index = pd.to_datetime(raw.index)
        return df
    except Exception as err:
        print(f"  [warn] water_level fetch failed: {err}")
        return pd.DataFrame({"wl": []}, index=pd.to_datetime([]))


def coops_tide_predictions_hourly(start_str, end_str, station_id, datum="MLLW"):
    """Fetch NOAA hourly tide predictions as a WL fallback when observations are unavailable."""
    try:
        stn = Station(id=station_id)
        raw = stn.get_data(
            begin_date=start_str,
            end_date=end_str,
            product="predictions",
            datum=datum,
            units="metric",
            time_zone="gmt",
            interval="h",
        )
    except TypeError:
        # Older noaa_coops versions may not accept interval for predictions.
        try:
            stn = Station(id=station_id)
            raw = stn.get_data(
                begin_date=start_str,
                end_date=end_str,
                product="predictions",
                datum=datum,
                units="metric",
                time_zone="gmt",
            )
        except Exception as err:
            print(f"  [warn] tide predictions fetch failed: {err}")
            return pd.DataFrame({"wl": []}, index=pd.to_datetime([]))
    except Exception as err:
        print(f"  [warn] tide predictions fetch failed: {err}")
        return pd.DataFrame({"wl": []}, index=pd.to_datetime([]))

    if raw is None or len(raw) == 0:
        return pd.DataFrame({"wl": []}, index=pd.to_datetime([]))

    if "v" in raw.columns:
        vals = pd.to_numeric(raw["v"], errors="coerce")
    else:
        vals = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    df = pd.DataFrame({"wl": vals.values})
    df.index = pd.to_datetime(raw.index, errors="coerce")
    return df[~df.index.isna()]


def coops_wind_native(start_str, end_str, station_id):
    try:
        stn = Station(id=station_id)
        raw = stn.get_data(begin_date=start_str, end_date=end_str, product="wind", units="metric", time_zone="gmt")
        df = pd.DataFrame({"speed": raw.s, "direction": raw.d})
        df.index = pd.to_datetime(raw.index)
        return df
    except Exception as err:
        print(f"  [warn] wind fetch failed: {err}")
        return pd.DataFrame({"speed": [], "direction": []}, index=pd.to_datetime([]))


def coops_scalar(product, start_str, end_str, station_id, colname):
    try:
        stn = Station(id=station_id)
        raw = stn.get_data(begin_date=start_str, end_date=end_str, product=product, units="metric", time_zone="gmt")
        df = pd.DataFrame({colname: raw.v})
        df.index = pd.to_datetime(raw.index)
        return df
    except Exception as err:
        print(f"  [warn] {product} fetch failed: {err}")
        return pd.DataFrame({colname: []}, index=pd.to_datetime([]))


def load_salinity_hourly(year, site):
    idx = hourly_index_for_year(year)
    out = pd.DataFrame({"datetime": idx, "salinity": np.nan})
    try:
        iv, _ = nwis.get_iv(
            sites=site,
            parameterCd="00480",
            start=idx[0].strftime("%Y-%m-%d"),
            end=(idx[-1] + pd.Timedelta(hours=1)).strftime("%Y-%m-%d"),
        )
        cols = [c for c in iv.columns if "00480" in str(c)]
        if not cols:
            return out
        s = iv[[cols[0]]].rename(columns={cols[0]: "salinity"}).rename_axis("datetime").reset_index()
        return to_hourly_spline(s, "salinity", idx, roll_win=3)
    except Exception as err:
        print(f"  [warn] salinity fetch failed: {err}")
        return out


def quick_plot(df, value_col, title, ylabel, color="tab:blue"):
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.plot(df["datetime"], df[value_col], lw=0.8, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    return fig, ax


def _http_get_json(url, params, timeout=60):
    qs = urllib.parse.urlencode(params, doseq=True)
    full = f"{url}?{qs}"
    with urllib.request.urlopen(full, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw)


def load_openmeteo_hourly(year, lat, lon, timeout=60):
    idx = hourly_index_for_year(year)
    empty = pd.DataFrame({"datetime": idx, "humidity": np.nan, "air_temp": np.nan, "cloudiness": np.nan, "ghi": np.nan, "wind_speed": np.nan, "wind_direction": np.nan, "pressure": np.nan})
    
    # Use the window dates for the API request
    start, end = year_window(year)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(["temperature_2m", "relative_humidity_2m", "cloud_cover", "shortwave_radiation", 
                            "wind_speed_10m", "wind_direction_10m", "pressure_msl"]),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    try:
        data = _http_get_json(OPENMETEO_URL, params, timeout=timeout)
    except urllib.error.HTTPError as err:
        print(f"  [open-meteo] HTTP {err.code}: {err.reason}  (URL: {OPENMETEO_URL}?...)")
        return empty
    except urllib.error.URLError as err:
        print(f"  [open-meteo] network error: {err.reason}")
        return empty
    except Exception as err:
        print(f"  [open-meteo] {type(err).__name__}: {err}")
        return empty

    if "hourly" not in data:
        print(f"  [open-meteo] unexpected response: keys={list(data.keys())}")
        return empty

    h = data["hourly"]
    h_units = data.get("hourly_units", {})
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(h["time"]),
            "air_temp": pd.to_numeric(h["temperature_2m"], errors="coerce"),
            "humidity": pd.to_numeric(h["relative_humidity_2m"], errors="coerce"),
            "cloudiness": pd.to_numeric(h["cloud_cover"], errors="coerce"),
            "ghi": pd.to_numeric(h["shortwave_radiation"], errors="coerce"),
            "wind_speed": pd.to_numeric(h["wind_speed_10m"], errors="coerce"),
            "wind_direction": pd.to_numeric(h["wind_direction_10m"], errors="coerce"),
            "pressure": pd.to_numeric(h["pressure_msl"], errors="coerce"),  # Already in hPa
        }
    )
    # Open-Meteo defaults wind speed to km/h unless wind_speed_unit=ms is honored.
    if h_units.get("wind_speed_10m") == "km/h":
        df["wind_speed"] = df["wind_speed"] / 3.6
    _ensure_naive_datetime(df, "datetime")

    def _align(col):
        return to_hourly_spline(df[["datetime", col]], col, idx, roll_win=1)

    out = _align("humidity").merge(_align("air_temp"), on="datetime").merge(_align("cloudiness"), on="datetime").merge(_align("ghi"), on="datetime").merge(_align("wind_speed"), on="datetime").merge(_align("wind_direction"), on="datetime").merge(_align("pressure"), on="datetime")

    print(
        f"  [open-meteo] {year}: {len(df):,} rows fetched, annual means - "
        f"H={out['humidity'].mean():.1f}%, T={out['air_temp'].mean():.1f}C, "
        f"CC={out['cloudiness'].mean():.1f}%, GHI={out['ghi'].mean():.0f} W/m2, "
        f"Wind={out['wind_speed'].mean():.2f} m/s, P={out['pressure'].mean():.1f} hPa"
    )
    return out


def compute_solar_ghi_pvlib(year, lat, lon, cloud_pct_series):
    loc = Location(lat, lon, "UTC", 2)
    times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left", tz="UTC")
    clearsky = loc.get_clearsky(times, model="ineichen")
    naive_idx = times.tz_convert(None)
    cc = cloud_pct_series.copy()
    cc.index = pd.DatetimeIndex(cc.index)
    c = cc.reindex(naive_idx, method="nearest").values / 100.0
    c = np.clip(np.nan_to_num(c, nan=0.5), 0.0, 1.0)
    ghi = clearsky["ghi"].values * (1.0 - 0.75 * np.power(c, 3.4))
    return pd.DataFrame({"datetime": naive_idx, "ghi": ghi})


def _pull_salinity_single_site(site, year):
    idx = hourly_index_for_year(year)
    start = idx[0].strftime("%Y-%m-%d")
    end = (idx[-1] + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
    empty = pd.DataFrame({"datetime": idx, "salinity": np.nan})
    site = str(site)

    def _to_hourly_raw(df, value_col):
        s = df.set_index("datetime")[value_col] if "datetime" in df.columns else df[value_col]
        s.index = pd.to_datetime(s.index, errors="coerce")
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_convert("UTC").tz_localize(None)
        s = pd.to_numeric(s, errors="coerce")
        return s.resample("h").mean().reindex(idx).to_frame(name=value_col).rename_axis("datetime").reset_index()

    def _salinity_from_00480():
        try:
            iv, _ = nwis.get_iv(sites=site, parameterCd="00480", start=start, end=end)
            cols = [c for c in iv.columns if "00480" in str(c)] if iv is not None else []
            if cols:
                s = iv[[cols[0]]].rename(columns={cols[0]: "salinity"}).rename_axis("datetime").reset_index()
                s["salinity"] = pd.to_numeric(s["salinity"], errors="coerce")
                n = s["salinity"].notna().sum()
                if n >= 24:
                    hourly = _to_hourly_raw(s, "salinity")
                    got = hourly["salinity"].notna().sum()
                    print(f"    {site} 00480: {n:,} raw has {got:,} hourly non-NaN")
                    return hourly
        except Exception as err:
            print(f"    {site} 00480 error: {type(err).__name__}: {err}")
        return None

    def _salinity_from_00095():
        try:
            iv, _ = nwis.get_iv(sites=site, parameterCd="00095", start=start, end=end)
            cols = [c for c in iv.columns if "00095" in str(c)] if iv is not None else []
            if cols:
                s = iv[[cols[0]]].rename(columns={cols[0]: "spc"}).rename_axis("datetime").reset_index()
                spc = pd.to_numeric(s["spc"], errors="coerce")
                s["salinity"] = 0.00066 * spc
                n = s["salinity"].notna().sum()
                if n >= 24:
                    hourly = _to_hourly_raw(s[["datetime", "salinity"]], "salinity")
                    got = hourly["salinity"].notna().sum()
                    print(f"    {site} 00095 (PSU): {n:,} raw has {got:,} hourly non-NaN")
                    return hourly
        except Exception as err:
            print(f"    {site} 00095 error: {type(err).__name__}: {err}")
        return None

    prefer_cond = site in USGS_SALINITY_PREFER_CONDUCTANCE
    strategies = [_salinity_from_00095, _salinity_from_00480] if prefer_cond else [_salinity_from_00480, _salinity_from_00095]
    for strategy in strategies:
        hourly = strategy()
        if hourly is not None:
            return hourly

    print(f"    {site}: no usable salinity data")
    return empty


def load_salinity_with_backups(year, primary_site, backup_sites):
    print(f"  [salinity] primary: {primary_site}")
    combined = _pull_salinity_single_site(primary_site, year)
    total = len(combined)
    for site in backup_sites:
        remaining = combined["salinity"].isna().sum()
        if remaining == 0:
            break
        print(f"  [salinity] gap-fill from backup {site} ({remaining:,} NaN hours remaining)")
        backup = _pull_salinity_single_site(site, year)
        mask = combined["salinity"].isna() & backup["salinity"].notna()
        if mask.any():
            combined.loc[mask, "salinity"] = backup.loc[mask, "salinity"]
            print(f"    filled {mask.sum():,} hours from {site}")

    s = combined.set_index("datetime")["salinity"]
    s = pd.to_numeric(s, errors="coerce").interpolate(method="time", limit=6, limit_direction="both").clip(lower=0.0)
    combined = s.to_frame(name="salinity").rename_axis("datetime").reset_index()

    final = combined["salinity"].notna().sum()
    print(f"  [salinity] final: {final:,}/{total:,} hourly values ({100 * final / total:.1f}% coverage)")
    return combined


def _pull_water_temp_single_station(station_id, start_str, end_str):
    try:
        stn = Station(id=station_id)
        raw = stn.get_data(begin_date=start_str, end_date=end_str, product="water_temperature", units="metric", time_zone="gmt")
        df = pd.DataFrame({"water_temp": raw.v})
        df.index = pd.to_datetime(raw.index)
        return df
    except Exception as err:
        print(f"    {station_id} error: {type(err).__name__}: {err}")
        return pd.DataFrame({"water_temp": []}, index=pd.to_datetime([]))


def _pull_water_temp_single_site_usgs(site_id, start_str, end_str):
    def _to_iso_or_raw(d):
        ts = pd.to_datetime(d, errors="coerce")
        return ts.strftime("%Y-%m-%d") if pd.notna(ts) else str(d)

    def _extract_temp_df(raw_df):
        if raw_df is None or raw_df.empty:
            return pd.DataFrame({"water_temp": []}, index=pd.to_datetime([]))
        cols = [c for c in raw_df.columns if "00010" in str(c)]
        if not cols:
            cols = [c for c in raw_df.columns if np.issubdtype(raw_df[c].dtype, np.number)]
        if not cols:
            return pd.DataFrame({"water_temp": []}, index=pd.to_datetime([]))
        vals = pd.to_numeric(raw_df[cols[0]], errors="coerce")
        idx = pd.to_datetime(raw_df.index, errors="coerce")
        df = pd.DataFrame({"water_temp": vals.values}, index=idx)
        return df[~df.index.isna()]

    sid = str(site_id).strip()
    start_iso = _to_iso_or_raw(start_str)
    end_iso = _to_iso_or_raw(end_str)
    attempts = [
        ("get_iv", start_iso, end_iso),
        ("get_record", start_iso, end_iso),
        ("get_iv", str(start_str), str(end_str)),
    ]

    last_err = None
    for method, s_arg, e_arg in attempts:
        try:
            if method == "get_iv":
                raw, _ = nwis.get_iv(sites=sid, parameterCd="00010", start=s_arg, end=e_arg)
            else:
                raw = nwis.get_record(
                    sites=sid,
                    service="iv",
                    parameterCd="00010",
                    start=s_arg,
                    end=e_arg,
                )
            out = _extract_temp_df(raw)
            if not out.empty:
                return out
        except Exception as err:
            last_err = err
            continue

    if last_err is not None:
        print(f"    USGS {sid} error: {type(last_err).__name__}: {last_err}")
    return pd.DataFrame({"water_temp": []}, index=pd.to_datetime([]))


def load_water_temp_with_backups(year, primary_usgs_site, backup_stations):
    idx = hourly_index_for_year(year)
    s_str, e_str = f"{year}0101", f"{year + 1}0101"
    print(f"  [water_temp] primary USGS: {primary_usgs_site}")
    raw = _pull_water_temp_single_site_usgs(primary_usgs_site, s_str, e_str)
    n_raw_primary = raw["water_temp"].notna().sum() if not raw.empty else 0
    print(f"    {primary_usgs_site}: {n_raw_primary:,} raw values")

    if not raw.empty:
        combined = hourly_ma_spline(raw[["water_temp"]], "water_temp", win=3)
        combined = combined.set_index("datetime").reindex(idx).reset_index().rename(columns={"index": "datetime"})
        combined["water_temp_source"] = np.where(combined["water_temp"].notna(), str(primary_usgs_site), pd.NA)
    else:
        combined = pd.DataFrame(
            {
                "datetime": idx,
                "water_temp": np.nan,
                "water_temp_source": pd.Series([pd.NA] * len(idx), dtype="object"),
            }
        )

    for station in backup_stations:
        remaining = combined["water_temp"].isna().sum()
        if remaining == 0:
            break
        print(f"  [water_temp] gap-fill from NOAA backup {station} ({remaining:,} NaN remaining)")
        raw_b = _pull_water_temp_single_station(station, s_str, e_str)
        if raw_b.empty or raw_b["water_temp"].notna().sum() == 0:
            print(f"    {station}: no data")
            continue
        hourly_b = hourly_ma_spline(raw_b[["water_temp"]], "water_temp", win=3)
        hourly_b = hourly_b.set_index("datetime").reindex(idx).reset_index().rename(columns={"index": "datetime"})
        mask = combined["water_temp"].isna() & hourly_b["water_temp"].notna()
        if mask.any():
            combined.loc[mask, "water_temp"] = hourly_b.loc[mask, "water_temp"]
            combined.loc[mask, "water_temp_source"] = str(station)
            print(f"    filled {mask.sum():,} hours from {station}")

    final = combined["water_temp"].notna().sum()
    print(f"  [water_temp] final: {final:,}/{len(idx):,} hourly values ({100 * final / len(idx):.1f}% coverage)")
    return combined


def _resample_hourly_raw(df, value_col, hourly_index):
    s = df.set_index("datetime")[value_col] if "datetime" in df.columns else df[value_col]
    s.index = pd.to_datetime(s.index, errors="coerce")
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s.resample("h").mean().reindex(hourly_index).to_frame(name=value_col).rename_axis("datetime").reset_index()


def _contiguous_nan_ranges(series):
    is_nan = pd.Series(series).isna().values
    out, i, n = [], 0, len(is_nan)
    while i < n:
        if is_nan[i]:
            j = i
            while j < n and is_nan[j]:
                j += 1
            out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def load_water_level_with_backups(year, primary_station, backup_stations, datum="MLLW", gap_threshold_hours=24):
    idx = hourly_index_for_year(year)
    s_str, e_str = f"{year}0101", f"{year + 1}0101"

    def _to_hourly_raw(raw_df):
        if raw_df.empty or "wl" not in raw_df.columns:
            return pd.DataFrame({"datetime": idx, "wl": np.nan})
        s = raw_df["wl"].copy()
        s.index = pd.to_datetime(s.index, errors="coerce")
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_convert("UTC").tz_localize(None)
        return s.resample("h").mean().reindex(idx).to_frame(name="wl").rename_axis("datetime").reset_index()

    print(f"  [water_level] primary: {primary_station}")
    raw = coops_water_level_native(s_str, e_str, primary_station, datum=datum)
    combined = _to_hourly_raw(raw)
    n_pri = combined["wl"].notna().sum()
    print(f"    {primary_station}: {n_pri:,} / {len(idx):,} hourly values ({100 * n_pri / len(idx):.1f}%)")

    gaps = _contiguous_nan_ranges(combined["wl"])
    long_gaps = [(a, b) for a, b in gaps if (b - a + 1) > gap_threshold_hours]

    if long_gaps and backup_stations:
        total_long = sum(b - a + 1 for a, b in long_gaps)
        print(f"    {len(long_gaps)} gaps > {gap_threshold_hours}h ({total_long:,} h total) - trying backup stations")

        for backup_id in backup_stations:
            if not long_gaps:
                break
            print(f"  [water_level] backup: {backup_id}")
            raw_b = coops_water_level_native(s_str, e_str, backup_id, datum=datum)
            backup_df = _to_hourly_raw(raw_b)
            if backup_df["wl"].notna().sum() == 0:
                print(f"    {backup_id}: no data")
                continue

            overlap = combined["wl"].notna() & backup_df["wl"].notna()
            n_over = int(overlap.sum())
            if n_over < 24 * 7:
                print(f"    {backup_id}: overlap {n_over}h (need >=168) - cannot regress")
                continue
            x = backup_df.loc[overlap, "wl"].values
            y = combined.loc[overlap, "wl"].values
            a, b = np.polyfit(x, y, 1)
            r = float(np.corrcoef(x, y)[0, 1])
            rmse = float(np.sqrt(np.mean((y - (a * x + b)) ** 2)))
            print(f"    {backup_id}: overlap={n_over:,}h  r={r:+.3f}  primary ~= {a:+.3f}*backup + {b:+.3f}  (fit RMSE {rmse:.3f} m)")

            if r < 0.3:
                print("       r too low - skipping")
                continue

            new_long, total_filled = [], 0
            for gs, ge in long_gaps:
                sl = slice(gs, ge + 1)
                bv = backup_df["wl"].iloc[sl]
                tv = a * bv + b
                fillable = combined["wl"].iloc[sl].isna() & tv.notna()
                n_fill = int(fillable.sum())
                if n_fill > 0:
                    combined.loc[combined.index[sl][fillable], "wl"] = tv[fillable].values
                    total_filled += n_fill
                if combined["wl"].iloc[sl].isna().any():
                    new_long.append((gs, ge))
            print(f"       filled {total_filled:,} hours")
            long_gaps = new_long

    # If observations are missing, backfill gaps with NOAA tide predictions so
    # historical years without archived observed WL can still be modeled.
    remaining_nan = combined["wl"].isna().sum()
    if remaining_nan > 0:
        print(f"  [water_level] filling remaining {remaining_nan:,} NaN hours from NOAA tide predictions")
        pred = _to_hourly_raw(coops_tide_predictions_hourly(s_str, e_str, primary_station, datum=datum))
        fillable = combined["wl"].isna() & pred["wl"].notna()
        if fillable.any():
            combined.loc[fillable, "wl"] = pred.loc[fillable, "wl"]
            print(f"    prediction fill from {primary_station}: {int(fillable.sum()):,} hours")

    remaining_nan = combined["wl"].isna().sum()
    if remaining_nan > 0 and backup_stations:
        for backup_id in backup_stations:
            pred_b = _to_hourly_raw(coops_tide_predictions_hourly(s_str, e_str, backup_id, datum=datum))
            fillable = combined["wl"].isna() & pred_b["wl"].notna()
            if fillable.any():
                combined.loc[fillable, "wl"] = pred_b.loc[fillable, "wl"]
                print(f"    prediction fill from backup {backup_id}: {int(fillable.sum()):,} hours")
            if combined["wl"].isna().sum() == 0:
                break

    series = combined.set_index("datetime")["wl"]
    try:
        lp = _safe_lowpass(series, cutoff_hours=6.0, order=6)
    except Exception as err:
        print(f"    [warn] low-pass failed ({err}) - using unfiltered values")
        lp = series.copy()
    combined["wl_lp"] = lp.values

    try:
        combined["wl_lp"] = combined.set_index("datetime")["wl_lp"].interpolate(method="spline", order=3, limit_direction="both").values
    except Exception:
        combined["wl_lp"] = combined.set_index("datetime")["wl_lp"].interpolate(method="time", limit_direction="both").values

    n_final = combined["wl_lp"].notna().sum()
    print(f"  [water_level] final: {n_final:,} / {len(idx):,} hourly values ({100 * n_final / len(idx):.1f}%)")
    return combined[["datetime", "wl_lp"]]


def air_to_river_temp(air_temp_series, lag_days, smooth_days, tmin, tmax, offset):
    dt_h = air_temp_series.index.to_series().diff().dt.total_seconds().dropna().median() / 3600
    steps_per_day = 24.0 / dt_h
    smooth_steps = int(round(smooth_days * steps_per_day))
    lag_steps = int(round(lag_days * steps_per_day))

    smoothed = air_temp_series.rolling(smooth_steps, min_periods=1, center=False).mean()
    lagged = smoothed.shift(lag_steps) + offset
    river_temp = lagged.clip(lower=tmin, upper=tmax).bfill()
    return river_temp


def _fetch_noaa_station_meta(station_id):
    url = f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        st = payload.get("stations", [{}])[0] if isinstance(payload.get("stations"), list) else payload
        return {
            "source": "NOAA",
            "station_id": str(station_id),
            "name": st.get("name", f"NOAA {station_id}"),
            "lat": float(st.get("lat")),
            "lon": float(st.get("lng")),
        }
    except Exception as exc:
        print(f"[warn] NOAA metadata failed for {station_id}: {exc}")
        return None


def _fetch_usgs_site_meta(site_no):
    try:
        info, _ = nwis.get_info(sites=site_no)
        if info is None or info.empty:
            return None
        row = info.iloc[0]
        return {
            "source": "USGS",
            "station_id": str(site_no),
            "name": str(row.get("station_nm", f"USGS {site_no}")),
            "lat": float(row.get("dec_lat_va")),
            "lon": float(row.get("dec_long_va")),
        }
    except Exception as exc:
        print(f"[warn] USGS metadata failed for {site_no}: {exc}")
        return None


def _add_record(rec_map, rec, role):
    if rec is None:
        return
    key = (rec["source"], rec["station_id"])
    if key not in rec_map:
        rec_map[key] = {**rec, "roles": [role]}
    else:
        rec_map[key]["roles"].append(role)


def _hourly_noaa_wt(station_id, year):
    s_str, e_str = f"{year}0101", f"{year + 1}0101"
    raw = _pull_water_temp_single_station(station_id, s_str, e_str)
    if raw.empty or raw["water_temp"].notna().sum() == 0:
        idx = hourly_index_for_year(year)
        return pd.Series(np.nan, index=idx, dtype=float)
    h = hourly_ma_spline(raw[["water_temp"]], "water_temp", win=3)
    idx = hourly_index_for_year(year)
    return h.set_index("datetime")["water_temp"].reindex(idx)


def _hourly_usgs_wt(site_id, year):
    s_str, e_str = f"{year}0101", f"{year + 1}0101"
    raw = _pull_water_temp_single_site_usgs(site_id, s_str, e_str)
    if raw.empty or raw["water_temp"].notna().sum() == 0:
        idx = hourly_index_for_year(year)
        return pd.Series(np.nan, index=idx, dtype=float)
    h = hourly_ma_spline(raw[["water_temp"]], "water_temp", win=3)
    idx = hourly_index_for_year(year)
    return h.set_index("datetime")["water_temp"].reindex(idx)


def _hourly_usgs_sal(site_id, year):
    h = _pull_salinity_single_site(site_id, year)
    if h.empty:
        idx = hourly_index_for_year(year)
        return pd.Series(np.nan, index=idx, dtype=float)
    idx = hourly_index_for_year(year)
    return h.set_index("datetime")["salinity"].reindex(idx)


def _water_temp_usage_by_year(year):
    idx = hourly_index_for_year(year)
    source = pd.Series(pd.NA, index=idx, dtype="object")

    p = _hourly_usgs_wt(USGS_WATER_TEMP_SITE, year)
    source.loc[p.notna()] = str(USGS_WATER_TEMP_SITE)

    for sid in WT_BACKUP_STATIONS:
        b = _hourly_noaa_wt(sid, year)
        fill_mask = source.isna() & b.notna()
        source.loc[fill_mask] = str(sid)

    return source.value_counts(dropna=True).to_dict()


def _salinity_usage_by_year(year):
    idx = hourly_index_for_year(year)
    source = pd.Series(pd.NA, index=idx, dtype="object")

    p = _hourly_usgs_sal(USGS_SALINITY_SITE, year)
    source.loc[p.notna()] = str(USGS_SALINITY_SITE)

    for sid in USGS_SALINITY_BACKUPS:
        b = _hourly_usgs_sal(sid, year)
        fill_mask = source.isna() & b.notna()
        source.loc[fill_mask] = str(sid)

    return source.value_counts(dropna=True).to_dict()


def _wl_raw_hourly(station_id, year):
    idx = hourly_index_for_year(year)
    s_str, e_str = f"{year}0101", f"{year + 1}0101"
    raw = coops_water_level_native(s_str, e_str, station_id, datum=NOAA_DATUM)
    if raw.empty or "wl" not in raw.columns:
        return pd.Series(np.nan, index=idx, dtype=float)
    s = raw["wl"].copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s.resample("h").mean().reindex(idx)


def _water_level_usage_by_year(year):
    primary = _wl_raw_hourly(WL_STATION, year)
    combined = primary.copy()
    counts = {str(WL_STATION): int(primary.notna().sum())}

    gaps = _contiguous_nan_ranges(combined)
    long_gaps = [(a, b) for a, b in gaps if (b - a + 1) > int(WL_GAP_THRESHOLD_H)]

    for sid in WL_BACKUP_STATIONS:
        if not long_gaps:
            break
        backup = _wl_raw_hourly(sid, year)
        if backup.notna().sum() == 0:
            continue

        overlap = combined.notna() & backup.notna()
        n_over = int(overlap.sum())
        if n_over < 24 * 7:
            continue
        x = backup.loc[overlap].values
        y = combined.loc[overlap].values
        a, b0 = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(r) or r < 0.3:
            continue

        filled_this_station = 0
        new_long = []
        for gs, ge in long_gaps:
            sl = slice(gs, ge + 1)
            bv = backup.iloc[sl]
            can_fill = bv.notna().values
            if can_fill.any():
                trans = a * bv.values + b0
                block = combined.iloc[sl].values
                block[can_fill] = trans[can_fill]
                combined.iloc[sl] = block
                filled_this_station += int(can_fill.sum())
            still_nan = pd.isna(combined.iloc[sl]).values
            if still_nan.any():
                loc = np.where(still_nan)[0]
                new_long.append((gs + int(loc[0]), gs + int(loc[-1])))

        long_gaps = new_long
        if filled_this_station > 0:
            counts[str(sid)] = counts.get(str(sid), 0) + filled_this_station

    return counts


def assemble_year_stage1(year):
    idx = hourly_index_for_year(year)

    trib = trib_by_year[year]
    if not trib.empty:
        pieces = []
        for nm, g in trib.groupby("name", sort=False):
            gi = g[["datetime", "Q_cms"]].rename(columns={"Q_cms": "value"}).copy()
            gi["datetime"] = pd.to_datetime(gi["datetime"], errors="coerce")
            gi = to_hourly_spline(gi, "value", idx, roll_win=3)
            gi["name"] = nm
            pieces.append(gi.rename(columns={"value": "Q_cms"}))
        trib_hr = pd.concat(pieces, ignore_index=True).sort_values(["name", "datetime"]).reset_index(drop=True)
    else:
        trib_hr = pd.DataFrame({"datetime": idx, "Q_cms": np.nan, "name": np.nan})

    bcs = spillway_by_year[year]
    bcs_hr = to_hourly_spline(bcs.rename(columns={"Q_cms": "value"}), "value", idx, roll_win=3) if not bcs.empty else pd.DataFrame({"datetime": idx, "value": np.nan})
    bcs_hr = bcs_hr.rename(columns={"value": "Q_cms"})

    out = {
        "bcs": bcs_hr,
        "Trib": trib_hr,
        "WL": to_hourly_spline(wl_by_year[year], "wl_lp", idx, roll_win=1),
        "WIND": wind_to_hourly_uv(wind_by_year[year], idx, roll_win=3),
        "PRES": to_hourly_spline(pres_by_year[year], "pressure", idx, roll_win=1),
    }

    if INCLUDE_TEMPERATURE:
        met = openmeteo_by_year[year]
        solar = solar_by_year[year]
        if "water_temp_source" in watT_by_year[year].columns:
            wt_src = watT_by_year[year][["datetime", "water_temp_source"]].copy()
        else:
            wt_src = pd.DataFrame({"datetime": idx, "water_temp_source": pd.Series([pd.NA] * len(idx), dtype="object")})
        wt_src["datetime"] = pd.to_datetime(wt_src["datetime"], errors="coerce")
        wt_src = wt_src.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"], keep="last").set_index("datetime").reindex(idx).rename_axis("datetime").reset_index()

        out["AIR_T"] = to_hourly_spline(airT_by_year[year], "air_temp", idx, roll_win=1)
        out["WAT_T"] = to_hourly_spline(watT_by_year[year], "water_temp", idx, roll_win=1)
        out["WAT_T_SRC"] = wt_src
        out["HUM"] = met[["datetime", "humidity"]].copy()
        out["CLOUD"] = met[["datetime", "cloudiness"]].copy()
        out["AIR_T_MET"] = met[["datetime", "air_temp"]].copy()
        out["GHI"] = solar.copy()
        out["RIVER_T"] = rivT_by_year[year].copy()

    if INCLUDE_SALINITY:
        out["SAL"] = to_hourly_spline(sal_by_year[year], "salinity", idx, roll_win=1)

    return out


def fmt_month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def pad_lims(vmin, vmax, frac=0.05):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (0.0, 1.0)
    if vmin == vmax:
        d = 1.0 if vmax == 0 else abs(vmax) * 0.1
        return (vmin - d, vmax + d)
    s = vmax - vmin
    return (vmin - frac * s, vmax + frac * s)


def row_lims(series_list):
    a = [pd.to_numeric(s, errors="coerce") for s in series_list]
    return pad_lims(np.nanmin([x.min(skipna=True) for x in a]), np.nanmax([x.max(skipna=True) for x in a]), 0.05)


def wt_source_text(d):
    src_df = d.get("WAT_T_SRC", pd.DataFrame())
    if src_df is None or src_df.empty or "water_temp_source" not in src_df.columns:
        return "WT source: unknown"
    src = src_df["water_temp_source"].dropna().astype(str)
    if src.empty:
        return "WT source: none"
    total = len(src)
    usgs_n = int((src == USGS_WATER_TEMP_SITE).sum())
    noaa_n = total - usgs_n
    if noaa_n > 0:
        noaa_top = src[src != USGS_WATER_TEMP_SITE].value_counts().index[0]
        return f"WT source: USGS {USGS_WATER_TEMP_SITE} {100 * usgs_n / total:.1f}%, NOAA {100 * noaa_n / total:.1f}% (top {noaa_top})"
    return f"WT source: USGS {USGS_WATER_TEMP_SITE} 100.0%"


def _safe_sheet_name(name):
    bad = "[]:*?/\\"
    return "".join("_" if ch in bad else ch for ch in str(name))[:31]


def _wt_source_label_for_sheet(d):
    src_df = d.get("WAT_T_SRC", pd.DataFrame())
    if src_df is None or src_df.empty or "water_temp_source" not in src_df.columns:
        return "unknown"
    src = src_df["water_temp_source"].dropna().astype(str)
    if src.empty:
        return "none"
    total = len(src)
    usgs_n = int((src == USGS_WATER_TEMP_SITE).sum())
    noaa_n = total - usgs_n
    if noaa_n <= 0:
        return f"USGS:{USGS_WATER_TEMP_SITE}=100%"
    noaa_top = src[src != USGS_WATER_TEMP_SITE].value_counts().index[0]
    return f"USGS:{USGS_WATER_TEMP_SITE}=+ NOAA_backups"


def build_sheets(year, d):
    sheets = {}
    trib = d.get("Trib", pd.DataFrame())
    if trib is not None and not trib.empty:
        t = trib[["datetime", "name", "Q_cms"]].copy()
        t["variable"], t["units"], t["source"] = "tributary_discharge", "m3 s-1", "USGS_NWIS"
        t = t.rename(columns={"Q_cms": "value"})[["datetime", "name", "variable", "value", "units", "source"]]
        _ensure_naive_datetime(t, "datetime")
        sheets["tributaries"] = t

    def _single(name, var, units, src, df, value_col, sheet):
        if df is None or df.empty:
            sheets[sheet] = pd.DataFrame()
            return
        t = df[["datetime", value_col]].rename(columns={value_col: "value"}).copy()
        t["name"], t["variable"], t["units"], t["source"] = name, var, units, src
        t = t[["datetime", "name", "variable", "value", "units", "source"]]
        _ensure_naive_datetime(t, "datetime")
        sheets[sheet] = t

    _single("BCS", "spillway_discharge", "m3 s-1", "BCS_final_discharge_csv", d["bcs"], "Q_cms", "spillway")
    _single(f"{WL_STATION}", "water_level", "m", "NOAA_COOPS", d["WL"], "wl_lp", "water_level")
    _single(f"{MET_STATION}", "wind_speed", "m s-1", "OpenMeteo_ERA5", d["WIND"], "speed", "wind_speed")
    _single(f"{MET_STATION}", "air_pressure", "hPa", "OpenMeteo_ERA5", d["PRES"], "pressure", "air_pressure")
    if INCLUDE_TEMPERATURE:
        _single(f"{MET_STATION}", "air_temperature", "degC", "OpenMeteo_ERA5", d["AIR_T"], "air_temp", "air_temp_ERA5")
        _single(f"{USGS_WATER_TEMP_SITE}", "water_temperature", "degC", _wt_source_label_for_sheet(d), d["WAT_T"], "water_temp", "water_temp")
        _single("ERA5+Calc", "river_temperature", "degC", "Derived", d.get("RIVER_T", pd.DataFrame()), "river_temp", "river_temp")
        _single("ERA5", "humidity", "pct", "OpenMeteo_ERA5", d["HUM"], "humidity", "humidity")
        _single("ERA5", "cloudiness", "pct", "OpenMeteo_ERA5", d["CLOUD"], "cloudiness", "cloudiness")
        _single("ERA5", "solar_radiation", "W m-2", "OpenMeteo_ERA5", d["GHI"], "ghi", "solar_radiation")
    if INCLUDE_SALINITY:
        _single(f"{USGS_SALINITY_SITE}", "salinity", "psu", "USGS+backups", d["SAL"], "salinity", "salinity")
    return sheets


def _smooth_hourly_met(df, value_col, win=6):
    s = df.set_index("datetime")[value_col].copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = s.rolling(win, center=True, min_periods=1).mean()
    s = s.interpolate(method="time", limit_direction="both")
    return s.to_frame(name=value_col).rename_axis("datetime").reset_index()


def _reindex_hourly_nointerp(df, value_col, hourly_index):
    s = df.set_index("datetime")[value_col] if "datetime" in df.columns else df[value_col]
    s.index = pd.to_datetime(s.index, errors="coerce")
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = pd.to_numeric(s, errors="coerce").reindex(hourly_index)
    return s.to_frame(name=value_col).rename_axis("datetime").reset_index()


def _relative_to_specific_humidity(rh_pct, air_temp_c, pressure_hpa):
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 100.0)
    t = np.asarray(air_temp_c, dtype=float)
    p = np.asarray(pressure_hpa, dtype=float)
    es = 6.112 * np.exp((17.67 * t) / (t + 243.5))
    e = (rh / 100.0) * es
    denom = np.maximum(p - 0.378 * e, 1e-6)
    q = 0.622 * e / denom
    return np.clip(q, 0.0, 0.05)


def assemble_year_stage2(year):
    idx = hourly_index_for_year(year)

    trib = trib_by_year[year]
    if not trib.empty:
        pieces = []
        for nm, g in trib.groupby("name", sort=False):
            gi = g[["datetime", "Q_cms"]].rename(columns={"Q_cms": "value"}).copy()
            gi["datetime"] = pd.to_datetime(gi["datetime"], errors="coerce")
            gi = to_hourly_spline(gi, "value", idx, roll_win=3)
            gi["name"] = nm
            pieces.append(gi.rename(columns={"value": "Q_cms"}))
        trib_hr = pd.concat(pieces, ignore_index=True).sort_values(["name", "datetime"]).reset_index(drop=True)
    else:
        trib_hr = pd.DataFrame({"datetime": idx, "Q_cms": np.nan, "name": np.nan})

    bcs = spillway_by_year[year]
    bcs_hr = to_hourly_spline(bcs.rename(columns={"Q_cms": "value"}), "value", idx, roll_win=3) if not bcs.empty else pd.DataFrame({"datetime": idx, "value": np.nan})
    bcs_hr = bcs_hr.rename(columns={"value": "Q_cms"})

    out = {
        "bcs": bcs_hr,
        "Trib": trib_hr,
        "WL": _reindex_hourly_nointerp(wl_by_year[year], "wl_lp", idx),
        "WIND": wind_to_hourly_uv(wind_by_year[year], idx, roll_win=1),
        "PRES": _reindex_hourly_nointerp(pres_by_year[year], "pressure", idx),
    }

    if INCLUDE_TEMPERATURE:
        met = openmeteo_by_year[year]
        solar = solar_by_year[year]
        if "water_temp_source" in watT_by_year[year].columns:
            wt_src = watT_by_year[year][["datetime", "water_temp_source"]].copy()
        else:
            wt_src = pd.DataFrame({"datetime": idx, "water_temp_source": pd.Series([pd.NA] * len(idx), dtype="object")})
        wt_src["datetime"] = pd.to_datetime(wt_src["datetime"], errors="coerce")
        wt_src = wt_src.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"], keep="last").set_index("datetime").reindex(idx).rename_axis("datetime").reset_index()

        out["AIR_T"] = _reindex_hourly_nointerp(airT_by_year[year], "air_temp", idx)
        out["WAT_T"] = _reindex_hourly_nointerp(watT_by_year[year], "water_temp", idx)
        out["WAT_T_SRC"] = wt_src
        out["HUM"] = met[["datetime", "humidity"]].copy()
        out["QAIR"] = met[["datetime", "specific_humidity"]].copy()
        out["CLOUD"] = met[["datetime", "cloudiness"]].copy()
        out["AIR_T_MET"] = met[["datetime", "air_temp"]].copy()
        out["GHI"] = solar.copy()
        out["RIVER_T"] = rivT_by_year[year].copy()

    if INCLUDE_SALINITY:
        out["SAL"] = _reindex_hourly_nointerp(sal_by_year[year], "salinity", idx)

    return out


def _fmt_field(name, value):
    return f"{name:<{_FIELD_W}s}= {value}"


def _seconds_since(times, ref):
    return ((pd.DatetimeIndex(times) - ref).total_seconds()).round().astype("int64")


def make_timeseries(name, ref, times, quantity, unit, values):
    return {
        "name": name,
        "quantity": quantity,
        "unit": unit,
        "times_s": _seconds_since(times, ref).values,
        "values": np.asarray(values, dtype=float),
        "ref": ref,
    }


def write_bc(blocks, out_path):
    if not blocks:
        raise ValueError(f"No forcing blocks to write to {out_path}")
    ref = blocks[0]["ref"]
    ref_str = f"seconds since {ref:%Y-%m-%d %H:%M:%S}"
    lines = []
    for i, b in enumerate(blocks):
        if i > 0:
            lines.append("")
        lines.append("[forcing]")
        lines.append(_fmt_field("Name", b["name"]))
        lines.append(_fmt_field("Function", "timeseries"))
        lines.append(_fmt_field("Time-interpolation", "linear"))
        lines.append(_fmt_field("Quantity", "time"))
        lines.append(_fmt_field("Unit", ref_str))
        lines.append(_fmt_field("Quantity", b["quantity"]))
        lines.append(_fmt_field("Unit", b["unit"]))
        for t, v in zip(b["times_s"], b["values"]):
            lines.append(f"{int(t)}  {float(v)!r}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_tim(path, ref, times, columns, header_lines=None):
    t_min = ((pd.DatetimeIndex(times) - ref).total_seconds() / 60.0).to_numpy()
    value_cols = [np.asarray(v, float) for v in columns.values()]
    lines = []
    for i in range(len(t_min)):
        parts = [f"{float(t_min[i]):.7e}"] + [f"{float(c[i]):.7e}" for c in value_cols]
        lines.append(" ".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def wind_speed_dir_to_uv(speed_ms, dir_from_deg):
    s = np.asarray(speed_ms, dtype=float)
    th = np.deg2rad(np.asarray(dir_from_deg, dtype=float))
    return -s * np.sin(th), -s * np.cos(th)


def scalar_forcing_blocks(polyline, n_support_pts, ref, times, quantity, unit, values):
    return [
        make_timeseries(name=f"{polyline}_{i:04d}", ref=ref, times=times, quantity=quantity, unit=unit, values=values)
        for i in range(1, n_support_pts + 1)
    ]


def _smooth_temp_series(times, values, win_hours):
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(times))
    # Strong centered moving average to damp abrupt boundary-temperature drops.
    return s.rolling(int(win_hours), center=True, min_periods=1).mean().to_numpy()


def write_dfm_year(year, d):
    out_dir = DFM_OUT_ROOT / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = DFM_REF_DATE

    disch = []
    for t in RIVERS_INFO:
        g = d["Trib"][d["Trib"]["name"] == t["name"]].dropna(subset=["Q_cms"]).sort_values("datetime")
        if g.empty:
            print(f"  [skip] no discharge data for {t['name']}")
            continue
        disch.append(make_timeseries(name=t["polyline"], ref=ref, times=pd.DatetimeIndex(g["datetime"]), quantity="dischargebnd", unit="m3/s", values=g["Q_cms"].values))
    bcs = d["bcs"].dropna(subset=["Q_cms"]).sort_values("datetime")
    if bcs.empty:
        print(f"  [spillway] no data - Q=0 block for {SPILLWAY_BND['polyline']}")
        _idx = hourly_index_for_year(year)
        bcs = pd.DataFrame({"datetime": _idx, "Q_cms": np.zeros(len(_idx))})
    disch.append(make_timeseries(name=SPILLWAY_BND["polyline"], ref=ref, times=pd.DatetimeIndex(bcs["datetime"]), quantity="dischargebnd", unit="m3/s", values=bcs["Q_cms"].values))
    p = write_bc(disch, out_dir / DFM_FILES["discharge"])
    print(f"  wrote {p.name}  ({len(disch)} blocks)")

    wl = d["WL"].dropna(subset=["wl_lp"]).sort_values("datetime")
    wl_blocks = scalar_forcing_blocks(polyline=OPEN_BND["polyline"], n_support_pts=OPEN_BND["n_support_points"], ref=ref, times=pd.DatetimeIndex(wl["datetime"]), quantity="waterlevelbnd", unit="m", values=wl["wl_lp"].values)
    p = write_bc(wl_blocks, out_dir / DFM_FILES["waterlevel"])
    print(f"  wrote {p.name}  ({len(wl_blocks)} blocks)")

    if INCLUDE_TEMPERATURE:
        wt = d["WAT_T"].dropna(subset=["water_temp"]).sort_values("datetime")
        rt = d["RIVER_T"].dropna(subset=["river_temp"]).sort_values("datetime")
        rt_vals = _smooth_temp_series(rt["datetime"], rt["river_temp"].values, RIVER_TEMP_SMOOTH_HOURS)
        wt_vals = _smooth_temp_series(wt["datetime"], wt["water_temp"].values, OCEAN_TEMP_SMOOTH_HOURS)
        temp_blocks = []
        for t in RIVERS_INFO:
            temp_blocks += scalar_forcing_blocks(polyline=t["polyline"], n_support_pts=N_PTS_RIVER, ref=ref, times=pd.DatetimeIndex(rt["datetime"]), quantity="temperaturebnd", unit="degC", values=rt_vals)
        temp_blocks += scalar_forcing_blocks(polyline=SPILLWAY_BND["polyline"], n_support_pts=N_PTS_RIVER, ref=ref, times=pd.DatetimeIndex(rt["datetime"]), quantity="temperaturebnd", unit="degC", values=rt_vals)
        temp_blocks += scalar_forcing_blocks(polyline=OPEN_BND["polyline"], n_support_pts=OPEN_BND["n_support_points"], ref=ref, times=pd.DatetimeIndex(wt["datetime"]), quantity="temperaturebnd", unit="degC", values=wt_vals)
        p = write_bc(temp_blocks, out_dir / DFM_FILES["temperature"])
        print(f"  wrote {p.name}  ({len(temp_blocks)} blocks)")
    else:
        print("  [skip] temperature disabled")

    if INCLUDE_SALINITY:
        sl = d["SAL"].dropna(subset=["salinity"]).sort_values("datetime")
        river_sal_values = np.full(len(sl), RIVER_SALINITY_PPT, dtype=float)
        sal_blocks = []
        for t in RIVERS_INFO:
            if not t.get("has_salinity", True):
                continue
            sal_blocks += scalar_forcing_blocks(polyline=t["polyline"], n_support_pts=N_PTS_RIVER, ref=ref, times=pd.DatetimeIndex(sl["datetime"]), quantity="salinitybnd", unit="ppt", values=river_sal_values)
        if SPILLWAY_BND.get("has_salinity", True):
            sal_blocks += scalar_forcing_blocks(polyline=SPILLWAY_BND["polyline"], n_support_pts=N_PTS_RIVER, ref=ref, times=pd.DatetimeIndex(sl["datetime"]), quantity="salinitybnd", unit="ppt", values=river_sal_values)
        sal_blocks += scalar_forcing_blocks(polyline=OPEN_BND["polyline"], n_support_pts=OPEN_BND["n_support_points"], ref=ref, times=pd.DatetimeIndex(sl["datetime"]), quantity="salinitybnd", unit="ppt", values=sl["salinity"].values)
        p = write_bc(sal_blocks, out_dir / DFM_FILES["salinity"])
        print(f"  wrote {p.name}  ({len(sal_blocks)} blocks)")
    else:
        print("  [skip] salinity disabled")

    w = d["WIND"].sort_values("datetime")
    if not w.empty and w["speed"].notna().any():
        p = write_tim(out_dir / DFM_FILES["wind"], ref, pd.DatetimeIndex(w["datetime"]), {"wind_speed_m_per_s": w["speed"].values, "wind_direction_deg": w["direction"].values})
        print(f"  wrote {p.name}")

    pr = d["PRES"].sort_values("datetime")
    if not pr.empty and pr["pressure"].notna().any():
        p = write_tim(out_dir / DFM_FILES["pressure"], ref, pd.DatetimeIndex(pr["datetime"]), {"pressure_hPa": pr["pressure"].values})
        print(f"  wrote {p.name}")

    if INCLUDE_TEMPERATURE:
        met_times = d["HUM"]["datetime"].values
        air_t_met = d["AIR_T_MET"]["air_temp"].values
        humidity_pct = d["HUM"]["humidity"].values
        cloudiness = d["CLOUD"]["cloudiness"].values
        ghi = d["GHI"]["ghi"].values
        print(f"  meteo relative humidity: {humidity_pct.min():.1f}-{humidity_pct.max():.1f} %")
        p = write_tim(out_dir / DFM_FILES["meteo_composite"], ref, pd.DatetimeIndex(met_times), {"relative_humidity_pct": humidity_pct, "airtemp_degC": air_t_met, "cloudiness_pct": cloudiness, "solar_W_per_m2": ghi})
        print(f"  wrote {p.name}")
    else:
        print("  [skip] composite meteo disabled (temperature off)")

    return out_dir
