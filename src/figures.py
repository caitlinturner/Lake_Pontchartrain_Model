"""
Figure generation for Delft3D-FM Lake Pontchartrain forcing inputs.
All plotting functions accept kwargs for ylabel, title, color, etc.
and return the figure object so notebook cells can call plt.show() if desired.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


def plot_tributary_discharge(trib_by_year, years, rivers_info):
    """
    Plot tributary discharge for one or more years.

    Parameters:
    trib_by_year : dict
        Dict mapping year to DataFrame with columns [datetime, Q_cms, name]
    years : list
        Years to plot
    rivers_info : list
        Config list of tributary dicts with 'name' key
    
    Returns:
    fig : matplotlib.figure.Figure
    """
    import cmocean
    
    trib_names = [r["name"].strip() for r in rivers_info]
    trib_colors = dict(
        zip(trib_names, cmocean.cm.haline(np.linspace(0.1, 0.9, len(trib_names))))
    )
    
    fig, axes = plt.subplots(
        len(years), 1, figsize=(9, 2.2 * len(years)), squeeze=False
    )
    
    for ax, year in zip(axes.ravel(), years):
        df = trib_by_year[year]
        zero_only = []
        
        for name in trib_names:
            grp = df.loc[df["name"] == name].sort_values("datetime")
            if grp.empty:
                continue
            
            is_zero_only = np.allclose(grp["Q_cms"].to_numpy(), 0.0)
            zero_only.append(name) if is_zero_only else None
            
            ax.plot(
                grp["datetime"],
                grp["Q_cms"],
                lw=0.9,
                color=trib_colors[name],
                linestyle="--" if is_zero_only else "-",
                alpha=0.85 if is_zero_only else 1.0,
                label=name,
            )
        
        ax.set_title(f"Tributary discharge used for model input: {year}")
        ax.set_ylabel("Q (m³ s⁻¹)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        
        if zero_only:
            ax.text(
                0.01,
                0.97,
                "Zero-flow series: " + ", ".join(zero_only),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )
    
    axes[0, 0].legend(ncol=3, fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    
    return fig


def plot_meteorology(pres_by_year, wind_by_year, years):
    """
    Plot pressure and wind speed in a two-panel subplot for one or more years.
    
    Parameters:
    pres_by_year : dict
        Dict mapping year to DataFrame with columns [datetime, pressure]
    wind_by_year : dict
        Dict mapping year to DataFrame with columns [datetime, speed, direction]
    years : list
        Years to plot

    Returns:
    figs : list of matplotlib.figure.Figure
    """
    figs = []
    
    for year in years:
        fig, axes = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)

        axes[0].plot(
            pres_by_year[year]["datetime"],
            pres_by_year[year]["pressure"],
            lw=0.8,
            color="tab:purple",
        )
        axes[0].set_ylabel("Pressure (hPa)")

        axes[1].plot(
            wind_by_year[year]["datetime"],
            wind_by_year[year]["speed"],
            lw=0.8,
            color="tab:green",
        )
        axes[1].set_ylabel("Wind speed (m/s)")
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        axes[0].set_title(f"ERA5 pressure and wind speed ({year})")
        fig.tight_layout()
        figs.append(fig)
    
    return figs


def plot_era5(openmeteo_by_year, years, om_lat, om_lon):
    """
    Plot Open-Meteo ERA5 humidity, air temp, cloudiness, and solar radiation.
    
    Parameters:
    openmeteo_by_year : dict
        Dict mapping year to DataFrame with columns [datetime, humidity, air_temp, cloudiness, ghi]
    years : list
        Years to plot
    om_lat : float
        Open-Meteo latitude
    om_lon : float
        Open-Meteo longitude
    
    Returns:
    figs : list of matplotlib.figure.Figure
    """
    figs = []
    
    for year in years:
        d = openmeteo_by_year[year]
        fig, axes = plt.subplots(4, 1, figsize=(9, 6.5), sharex=True)
        
        axes[0].plot(d["datetime"], d["humidity"], lw=0.6, color="tab:blue")
        axes[0].set_ylabel("Humidity (%)")
        
        axes[1].plot(d["datetime"], d["air_temp"], lw=0.6, color="tab:red")
        axes[1].set_ylabel("Air T (°C)")
        
        axes[2].plot(d["datetime"], d["cloudiness"], lw=0.6, color="tab:gray")
        axes[2].set_ylabel("Cloudiness (%)")
        
        axes[3].plot(d["datetime"], d["ghi"], lw=0.4, color="tab:orange")
        axes[3].set_ylabel("GHI (W m⁻²)")
        axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        
        axes[0].set_title(f"Open-Meteo ERA5 at ({om_lat:.3f}, {om_lon:.3f}) ({year})")
        fig.tight_layout()
        figs.append(fig)
    
    return figs


def plot_summary_grid(
    data_by_year,
    years,
    rivers_info,
    wl_station,
    fig_dir,
    usgs_water_temp_site,
    pad_lims,
    row_lims,
    fmt_month_axis,
    wt_source_text,
    dpi=150,
    save_figure=False,
):
    """
    Generates 11-row × N-year summary figure showing all forcing variables
    
    Parameters:
    data_by_year : dict
        Dict mapping year → assembled forcing data dict
    years : list
        Years to plot
    rivers_info : list
        Config list of tributary dicts with 'name' key
    wl_station : str
        NOAA water level station ID
    fig_dir : Path
        Output directory for figure
    usgs_water_temp_site : str
        USGS water temp site ID (for legend)
    pad_lims : callable
        Function to pad axis limits (from hydromet_functions)
    row_lims : callable
        Function to compute row limits (from hydromet_functions)
    fmt_month_axis : callable
        Function to format month axis (from hydromet_functions)
    wt_source_text : callable
        Function to format water temp source text (from hydromet_functions)
    dpi : int, optional
        Figure DPI for rendering and optional save
    save_figure : bool, optional
        If True, write figure to fig_dir
    
    Returns:
    fig : matplotlib.figure.Figure
    """
    import cmocean
    
    # Set up colors for tributaries
    trib_names = [r["name"].strip() for r in rivers_info]
    trib_colors = dict(
        zip(trib_names, cmocean.cm.haline(np.linspace(0.1, 0.9, len(trib_names))))
    )
    
    # Format spec for each row: (key, ylabel, colname)
    row_specs = [
        ("Trib", "Tributaries\nQ (m³ s⁻¹)", "Q_cms"),
        ("bcs", "Diversion\nQ (m³ s⁻¹)", "Q_cms"),
        (
            "WL",
            f"Water level\n{wl_station} (m)",
            "wl_lp",
        ),
        ("WIND", "Wind speed\n(m s⁻¹)", "speed"),
        ("PRES", "Air pressure\n(hPa)", "pressure"),
        ("AIR_T", "Air temp\n(°C)", "air_temp"),
        ("WAT_T", "Water temp\n(°C)", "water_temp"),
        ("SAL", "Salinity\n(psu)", "salinity"),
        ("HUM", "Humidity\n(%)", "humidity"),
        ("CLOUD", "Cloudiness\n(%)", "cloudiness"),
        ("GHI", "Solar\n(W m⁻²)", "ghi"),
    ]
    
    fmt4 = FuncFormatter(lambda x, _pos: "{:.4g}".format(x) if np.isfinite(x) else "")
    
    n_rows, n_cols = len(row_specs), len(years)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 11),
        dpi=dpi,
        sharex="col",
        squeeze=False,
    )
    
    for r_idx, (key, ylab, colname) in enumerate(row_specs):
        series_for_row = []
        
        for j, year in enumerate(years):
            d = data_by_year[year]
            ax = axs[r_idx, j]
            
            if key == "Trib":
                zero_only = []
                if not d["Trib"].empty:
                    for nm in trib_names:
                        grp = d["Trib"].loc[d["Trib"]["name"] == nm].dropna(
                            subset=[colname]
                        ).sort_values("datetime")
                        if grp.empty:
                            continue
                        
                        is_zero_only = np.allclose(
                            pd.to_numeric(grp[colname], errors="coerce")
                            .fillna(0.0)
                            .to_numpy(),
                            0.0,
                        )
                        if is_zero_only:
                            zero_only.append(nm)
                        
                        ax.plot(
                            grp["datetime"],
                            grp[colname],
                            lw=0.9,
                            color=trib_colors.get(nm, "k"),
                            linestyle="--" if is_zero_only else "-",
                            alpha=0.85 if is_zero_only else 1.0,
                            label=nm,
                        )
                
                if zero_only:
                    ax.text(
                        0.01,
                        0.97,
                        "Zero-flow: " + ", ".join(zero_only),
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=7,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                    )
                
                if j == 0:
                    ax.set_ylabel(ylab)
                    ax.legend(
                        ncol=2,
                        frameon=False,
                        loc="upper left",
                        bbox_to_anchor=(0.04, 1.02),
                        borderaxespad=0.0,
                    )
                
                series_for_row.append(
                    d["Trib"][colname] if not d["Trib"].empty else pd.Series([np.nan])
                )
            else:
                if key in d and not d[key].empty and colname in d[key].columns:
                    ax.plot(d[key]["datetime"], d[key][colname], lw=0.9, color="k")
                    series_for_row.append(d[key][colname])
                else:
                    series_for_row.append(pd.Series([np.nan]))
                
                if j == 0:
                    ax.set_ylabel(ylab)
                
                if key == "WAT_T":
                    ax.text(
                        0.01,
                        0.97,
                        wt_source_text(d),
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=7,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                    )
            
            if r_idx == 0:
                ax.set_title(f"Year: {year}")
            
            ax.yaxis.set_major_formatter(fmt4)
        
        # Compute row limits and apply
        yl, yh = row_lims(series_for_row)
        if np.isfinite(yl) and np.isfinite(yh):
            for j in range(n_cols):
                axs[r_idx, j].set_ylim(yl, yh)
    
    # Format axes
    for j in range(n_cols):
        for i in range(n_rows - 1):
            axs[i, j].tick_params(labelbottom=False)
        fmt_month_axis(axs[-1, j])
    
    for i in range(n_rows):
        for j in range(1, n_cols):
            axs[i, j].tick_params(labelleft=False)
            axs[i, j].set_ylabel("")
    
    for ax in axs.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    fig.align_ylabels(axs[:, 0])
    plt.tight_layout()
    
    # Save if requested
    if save_figure:
        fig_name = "hydromet_" + "_".join(str(y) for y in years) + ".png"
        fig.savefig(fig_dir / fig_name, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved: {fig_dir / fig_name}")
    
    return fig
