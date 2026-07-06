"""Validation helpers for Lake Pontchartrain DFM model.
Handles downloading observations and computing validation statistics.
"""

from __future__ import annotations

import socket

import gsw
import numpy as np
import pandas as pd
from dataretrieval import nwis
from noaa_coops import Station
from scipy.signal import butter, filtfilt


## Constants
FT2M = 0.3048
USGS_TIMEOUT_MAX_ATTEMPTS = 3


def _is_timeout_error(err: Exception) -> bool:
    """Return True when an exception indicates a network timeout."""
    if isinstance(err, (TimeoutError, socket.timeout)):
        return True
    msg = f"{type(err).__name__}: {err}".lower()
    timeout_markers = ("timeout", "timed out", "readtimeout", "connecttimeout")
    return any(marker in msg for marker in timeout_markers)


## Data Type Conversion & Preprocessing
def _to_float(series):
    """Coerce a Series to float64, turning any non-numeric entries into NaN.
    USGS sometimes returns sentinel strings. this fixes that.
    """
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _to_naive_utc_indexed(series_or_df):
    """Convert to naive UTC-indexed, dropping rows with invalid index."""
    obj = series_or_df.copy()
    idx = pd.to_datetime(obj.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    obj.index = idx
    return obj[~obj.index.isna()]


def _fmt_day(d):
    """Format a datetime as YYYYMMDD string for web APIs."""
    return d.strftime("%Y-%m-%d")




## Statistics Computation
def compute_stats(obs, model):
    """Willmott d, Pearson r, and RMSE. Drops NaN pairs, handles edge cases.
    Returns:
    tuple
        (d, r, rmse, n, reason) where reason is a string explaining why
        stats could not be computed (empty if successful).
    """
    obs = _to_float(obs)
    model = _to_float(model)
    pair = pd.DataFrame({"obs": obs, "model": model}).dropna()
    n = len(pair)
    if n < 2:
        return np.nan, np.nan, np.nan, n, f"only {n} paired sample(s) — need ≥2"
    o = pair["obs"].values
    m = pair["model"].values

    # Willmott's d
    o_mean = o.mean()
    num = np.sum((m - o) ** 2)
    den = np.sum((np.abs(m - o_mean) + np.abs(o - o_mean)) ** 2)
    d = 1.0 - num / den if den > 0 else np.nan

    # Pearson r, undefined if either series has zero variance
    if o.std() == 0 or m.std() == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(o, m)[0, 1])
    rmse = float(np.sqrt(np.mean((m - o) ** 2)))
    return float(d), r, rmse, n, ""



## Observation Preprocessing by Variable Type
def hourly_ma_spline(series, win=3):
    """Hourly moving average + spline interpolation."""
    s = _to_naive_utc_indexed(_to_float(series).to_frame(name="value"))["value"]
    s = s.resample("h").mean().rolling(win, center=True, min_periods=1).mean()
    try:
        s = s.interpolate(method="spline", order=3, limit_direction="both")
    except Exception:
        s = s.interpolate(method="time", limit_direction="both")
    return s


def lowpass_6h_hourly(series, cutoff_hours=6.0, order=6):
    """6-hour Butterworth low-pass filter for water level tidal data."""
    s = _to_naive_utc_indexed(_to_float(series).to_frame(name="value"))["value"]

    if s.empty:
        return pd.Series(dtype="float64")
    s = s.interpolate(method="time", limit_direction="both")

    try:
        dt_seconds = s.index.to_series().diff().dt.total_seconds().dropna().median()
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            dt_seconds = 360.0
    except Exception:
        dt_seconds = 360.0

    wn = max(min(2.0 * dt_seconds / (cutoff_hours * 3600.0), 0.99), 1e-6)
    b, a = butter(order, wn, btype="low")
    edge = 3 * (max(len(a), len(b)) - 1)

    if len(s) <= edge + 1:
        win = max(int(round((cutoff_hours * 3600.0) / dt_seconds)), 3)
        filtered = s.rolling(win, center=True, min_periods=1).mean()
    else:
        filtered = pd.Series(filtfilt(b, a, s.values, method="pad"), index=s.index)
    return filtered.resample("h").mean()


def preprocess_obs_series(obs_series, variable):
    """Preprocess observations by variable type."""
    if len(obs_series) == 0:
        return pd.Series(dtype="float64")
    if variable == "wl":
        return lowpass_6h_hourly(obs_series)
    if variable in {"sal", "temp"}:
        return hourly_ma_spline(obs_series, win=3)
    return (
        _to_naive_utc_indexed(_to_float(obs_series).to_frame(name="value"))["value"]
        .resample("h")
        .mean()
    )


def align_model_obs(model_series, obs_series, variable):
    """Resample model to hourly, preprocess observations by variable, then pair.
    
    Forces float dtype on both sides so downstream metrics never see mixed types.
    """
    if len(model_series) == 0 or len(obs_series) == 0:
        return pd.DataFrame(columns=["model", "obs"])
    m = (
        _to_naive_utc_indexed(_to_float(model_series).to_frame(name="value"))["value"]
        .resample("h")
        .mean()
    )
    o = preprocess_obs_series(obs_series, variable)
    return pd.concat({"model": m, "obs": o}, axis=1).dropna()


## NOAA CO-OPS Downloaders
def _fetch_noaa(station_id, product, start, end, datum=None):
    """Generic NOAA CO-OPS product at its native service timestep."""
    stn = Station(id=station_id)
    kwargs = dict(
        begin_date=start.strftime("%Y%m%d"),
        end_date=(end + pd.Timedelta(days=1)).strftime("%Y%m%d"),
        product=product,
        units="metric",
        time_zone="gmt",
    )

    if datum is not None:
        kwargs["datum"] = datum
    raw = stn.get_data(**kwargs)
    return pd.Series(
        pd.to_numeric(raw["v"], errors="coerce").values,
        index=pd.to_datetime(raw.index),
    ).loc[start:end]


def fetch_noaa_wl(station_id, start, end, datum="MLLW"):
    """NOAA CO-OPS native water level (m) in the given datum."""
    s = _fetch_noaa(station_id, "water_level", start, end, datum=datum)
    return pd.DataFrame({"wl": s})


def fetch_noaa_water_temp(station_id, start, end):
    """NOAA CO-OPS water temperature (°C)."""
    s = _fetch_noaa(station_id, "water_temperature", start, end)
    return pd.DataFrame({"temp": s})



## USGS NWIS Downloaders
def _nwis_raw(site, param, start, end):
    """USGS NWIS instantaneous values at native cadence, tz-stripped."""
    last_error = None
    for attempt in range(1, USGS_TIMEOUT_MAX_ATTEMPTS + 1):
        try:
            raw = nwis.get_record(
                sites=site,
                service="iv",
                parameterCd=param,
                start=_fmt_day(start),
                end=_fmt_day(end + pd.Timedelta(days=1)),
            )
            break
        except Exception as err:
            if _is_timeout_error(err) and attempt < USGS_TIMEOUT_MAX_ATTEMPTS:
                print(
                    f"  [warn] USGS timeout for site {site}, param {param}; "
                    f"retrying ({attempt}/{USGS_TIMEOUT_MAX_ATTEMPTS})"
                )
                last_error = err
                continue
            raise
    else:
        raise last_error

    idx = pd.to_datetime(raw.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = pd.Series(pd.to_numeric(raw[param], errors="coerce").values, index=idx)
    return s.loc[start:end]


def fetch_usgs_wl(site, start, end):
    """USGS gauge height (parameter 00065) in m."""
    s = _nwis_raw(site, "00065", start, end) * FT2M
    return pd.DataFrame({"wl": s})


def fetch_usgs_water_temp(site, start, end):
    """USGS water temperature (parameter 00010) in °C."""
    s = _nwis_raw(site, "00010", start, end)
    return pd.DataFrame({"temp": s})


def fetch_usgs_salinity_direct(site, start, end):
    """USGS parameter 00480 (practical salinity, psu) direct."""
    s = _nwis_raw(site, "00480", start, end)
    return pd.DataFrame({"sal": s})


def fetch_usgs_salinity_from_conductance(site, start, end):
    """Practical salinity computed from USGS conductance + temp + depth via gsw.
    
    Uses params 00095 (conductance, μS/cm), 00010 (temperature, °C), and
    00065 (gauge height, ft → m). Matches the Pearl River path in
    validation_2021_salinity.py.
    """
    cond = _nwis_raw(site, "00095", start, end)  # μS/cm
    temp = _nwis_raw(site, "00010", start, end)  # °C
    depth = _nwis_raw(site, "00065", start, end) * FT2M  # m
    df = pd.concat({"cond": cond, "temp": temp, "depth": depth}, axis=1)
    df = df.interpolate(limit=3)

    # SP_from_C expects conductance in mS/cm, temp in °C, pressure in dbar.
    # For shallow water, depth(m) ≈ pressure(dbar).
    sal = gsw.SP_from_C(df["cond"] / 1000.0, df["temp"], df["depth"])
    return pd.DataFrame({"sal": sal.values}, index=df.index)



## Validation Processing
def validate_variable(
    stations_config,
    model_df,
    cal_start,
    cal_end,
    offsets_dict,
    variable_type,
):
    """Process validation for a single variable across multiple stations.
    
    Parameters:
    stations_config : list of dict
        List of station dicts with keys: label, model_col, fetcher, offset_key
    model_df : pd.DataFrame
        Model output with columns being station names, index being datetime
    cal_start : pd.Timestamp
        Validation window start
    cal_end : pd.Timestamp
        Validation window end
    offsets_dict : dict
        Dict mapping offset_key → offset value
    variable_type : str
        One of "wl", "sal", "temp" — used for preprocessing
    
    Returns:
    list of dict
        Results, one per station, with keys: label, model_col, fetcher, offset_key,
        paired (DataFrame), d, r, rmse, bias_hint, n, applied_offset
    """
    results = []

    for station in stations_config:
        label = station["label"]
        print(f"\n{label}")
        
        try:
            obs = station["fetcher"]()
            obs_col = list(obs.columns)[0]  # First column name
            n_ok = obs[obs_col].notna().sum()
            print(f"  fetched obs: {n_ok} raw values")
        except Exception as err:
            print(f"  [error] obs fetch failed: {type(err).__name__}: {err}")
            obs_col = list(station["fetcher"]().columns)[0]
            obs = pd.DataFrame({obs_col: pd.Series(dtype=float)})

        model_col = station["model_col"]
        if model_col not in model_df.columns:
            print(f"  [error] model column '{model_col}' not in history file")
            model_raw = pd.Series(dtype=float, name=model_col)
        else:
            model_raw = model_df[model_col]

        offset = offsets_dict[station["offset_key"]]
        model_adj = model_raw - offset
        paired = align_model_obs(model_adj, obs[obs_col], variable_type)

        if paired.empty:
            d = r = rmse = bias_hint = np.nan
            n = 0
            reason = "no paired samples — obs fetch empty or no overlap with model window"
        else:
            d, r, rmse, n, reason = compute_stats(paired["obs"], paired["model"])
            bias_hint = float((paired["model"] + offset - paired["obs"]).mean())

        result = {
            **station,
            "paired": paired,
            "d": d,
            "r": r,
            "rmse": rmse,
            "bias_hint": bias_hint,
            "n": n,
            "applied_offset": offset,
        }
        results.append(result)

        if reason:
            print(f"  [stats unavailable] {reason}")
            print(f"  | n_pairs={n}  bias_hint={bias_hint:+.3f}  offset={offset:+.3f}")
        else:
            print(
                f"  d={d:.2f}  r={r:.2f}  RMSE={rmse:.3f}  n={n}  "
                f"| offset={offset:+.3f}  bias_hint={bias_hint:+.3f}"
            )

    return results


# ============================================================================
# Model Data Loading
# ============================================================================


def load_model_variable(
    ds,
    station_names,
    variable_name,
    time_shift_hours,
    val_start,
    val_end,
):
    """Extract variable for all stations, apply time shift, slice to validation window.

    Parameters
    ----------
    ds : xr.Dataset
        Xarray dataset from FlowFM_his.nc
    station_names : pd.Series or list
        Station name strings matching column names in ds[variable_name]
    variable_name : str
        Variable name in ds (e.g., 'waterlevel', 'salinity', 'temperature')
    time_shift_hours : int or float
        Hours to shift model time (artifact for timezone correction)
    val_start : pd.Timestamp
        Validation window start
    val_end : pd.Timestamp
        Validation window end

    Returns
    -------
    pd.DataFrame
        DataFrame with stations as columns, datetime index, sliced to window
    """
    arr = ds[variable_name].to_pandas()
    if hasattr(station_names, 'values'):
        station_names = station_names.values
    arr.columns = station_names
    arr.index = pd.to_datetime(arr.index) + pd.Timedelta(hours=time_shift_hours)
    return arr.loc[val_start:val_end]
