## 1) Import packages 
import subprocess
import sys
from pathlib import Path

# Optional one-time install for video export only. #Keep it false for now
# Keep this False for normal use.
INSTALL_VIDEO_DEPS = False
if INSTALL_VIDEO_DEPS:
    # Needed so Matplotlib can find ffmpeg for MP4 writing.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "imageio-ffmpeg"])

# If matplotlib is broken in the active interpreter, fail with clear guidance.
try:
    import cmocean
    import matplotlib.pyplot as plt
except Exception as exc:
    msg = str(exc)
    broken_matplotlib = (
        isinstance(exc, ModuleNotFoundError)
        and getattr(exc, "name", "") == "matplotlib.backends.registry"
    ) or (isinstance(exc, AttributeError) and "get_data_path" in msg)
    if not broken_matplotlib:
        raise
    raise RuntimeError(
        "Matplotlib appears broken in this Python environment. "
        "Katie can try to fix "
        "`pip install --upgrade --force-reinstall matplotlib cmocean`."
    ) from exc

import xarray as xr

try:
    import xugrid as xu
except ImportError as exc:
    raise ImportError(
        "xugrid is required for plotting unstructured-grid NetCDF files. "
        "Install with pip install xugrid"
    ) from exc







## 2) Set file path
# Paste your full NetCDF path here:
model_nc = Path(r"c:\Users\hydro\Lake_Pontchartrain_Model\outputs\shared\lp_model_spatial_window_20130601_20131001_ugrid.nc")
if not model_nc.exists():
    raise FileNotFoundError(f"NetCDF file not found: {model_nc}")


## 3) Rename Variables
var_name_map = {
    "mesh2d_ucmag": "velocity_magnitude",
    "mesh2d_s1": "water_level",
    "mesh2d_tem1": "temperature",
    "mesh2d_sa1": "salinity",
}

# Put timesteps in better format
with xr.open_dataset(model_nc) as ds_selected:
    time_str_all = ds_selected.time.dt.strftime("%m-%d-%Y %H:%M:%S")

    t_start = time_str_all.isel(time=0).item()
    t_end = time_str_all.isel(time=-1).item()
    n_timesteps = ds_selected.sizes["time"]

print("\nDataset:", model_nc.name)
print("Time range:", t_start, "to", t_end)
print("Timesteps:", n_timesteps)


## 4) Create subplot preview figures
# Set per-variable colormap and colorbar limits (vmin/vmax).
plot_specs = [
    ("velocity_magnitude", "Velocity Magnitude $(m s^{-1})$", cmocean.cm.amp, 0.0, 1.2),
    ("water_level", "Water Level (m)", cmocean.cm.deep, -0.20, 1.0),
    ("temperature", "Temperature (°C)", cmocean.cm.thermal, 24.0, 34.0),
    ("salinity", "Salinity (PSU)", cmocean.cm.haline, 0.0, 18.0),
]

# Choose the timestep to preview.
preview_timestep = 0
t_str = time_str_all.isel(time=preview_timestep).item()
print("Plot timestamp:", t_str)
uds = xu.open_dataset(model_nc).rename(var_name_map)

# Font-size controls
title_fontsize = 9
axis_label_fontsize = 9
tick_fontsize = 8
cbar_label_fontsize = 9
cbar_tick_fontsize = 8

# Create Figure
fig, axes = plt.subplots(2, 2, figsize=(12, 5.5), dpi=200)
axes = axes.ravel()
for ax, (var_name, common_name, cmap, vmin, vmax) in zip(axes, plot_specs):
    quad = uds[var_name].isel(time=preview_timestep).ugrid.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={"label": common_name},
    )
    cbar = quad.colorbar
    cbar.set_label(common_name, fontsize=cbar_label_fontsize)
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

    ax.set_aspect("equal")
    ax.set_title(f"Date: {t_str}", fontsize=title_fontsize)
    ax.set_xlabel("Longitude (°)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Latitude (°)", fontsize=axis_label_fontsize)
    ax.set_aspect("equal")
    ax.set_facecolor("whitesmoke")
    ax.tick_params(labelsize=tick_fontsize)

plt.tight_layout()
plt.subplots_adjust(hspace=0.35)
plt.show()


## 5) Plot salinity at one timestep
preview_timestep = min(12, n_timesteps - 1)
t_str = time_str_all.isel(time=preview_timestep).item()
print("Plot timestamp:", t_str)

fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=200)
quad = uds["salinity"].isel(time=preview_timestep).ugrid.plot(
    ax=ax,
    cmap=cmocean.cm.haline,
    vmin=0.0,
    vmax=18.0,
    add_colorbar=True,
    cbar_kwargs={"label": "Salinity (PSU)"},
)

cbar = quad.colorbar
cbar.set_label("Salinity (PSU)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title(f"Date: {t_str}", fontsize=9)
ax.set_xlabel("Longitude (°)", fontsize=9)
ax.set_ylabel("Latitude (°)", fontsize=9)
ax.set_aspect("equal")
ax.set_facecolor("whitesmoke")
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()


## 6) Create Video (optional)
# # This may be broken. I have been trying to fix it.
# # Optional one-time install for video export only.
# # Keep this False for normal use.
# INSTALL_VIDEO_DEPS = False
# if INSTALL_VIDEO_DEPS:
#     # Needed so Matplotlib can find ffmpeg for MP4 writing.
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "imageio-ffmpeg"])

# # Choose which variable to animate: salinity, temperature, velocity, water_level
# movie_var = "temperature"  # Change this to the variable you want to animate

# movie_specs = {
#     "velocity": {
#         "var_name": "velocity_magnitude",
#         "label": "Velocity Magnitude $(m s^{-1})$",
#         "cmap": cmocean.cm.amp,
#         "vmin": 0.0,
#         "vmax": 0.8,
#         "file_tag": "velocity",
#     },
#     "water_level": {
#         "var_name": "water_level",
#         "label": "Water Level (m)",
#         "cmap": cmocean.cm.deep,
#         "vmin": -0.20,
#         "vmax": 1.0,
#         "file_tag": "water_level",
#     },
#     "temperature": {
#         "var_name": "temperature",
#         "label": "Temperature (°C)",
#         "cmap": cmocean.cm.thermal,
#         "vmin": 24.0,
#         "vmax": 34.0,
#         "file_tag": "temperature",
#     },
#     "salinity": {
#         "var_name": "salinity",
#         "label": "Salinity (PSU)",
#         "cmap": cmocean.cm.haline,
#         "vmin": 0.0,
#         "vmax": 18.0,
#         "file_tag": "salinity",
#     },
# }

# if movie_var not in movie_specs:
#     raise ValueError(
#         f"movie_var must be one of {list(movie_specs.keys())}, got '{movie_var}'."
#     )

# spec = movie_specs[movie_var]
# var_name = spec["var_name"]
# label = spec["label"]
# cmap = spec["cmap"]
# vmin = spec["vmin"]
# vmax = spec["vmax"]
# file_tag = spec["file_tag"]

# # Try to point Matplotlib to bundled ffmpeg if imageio-ffmpeg is installed.
# ffmpeg_ready = False
# try:
#     import imageio_ffmpeg

#     plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
#     ffmpeg_ready = True
# except Exception:
#     ffmpeg_ready = False

# # Movie settings
# start_idx = 0
# end_idx = n_timesteps - 1
# frame_step = 6          # change this to control timestep spacing between frames
# fps = 6                 # playback speed
# bitrate = 1800

# # Build frame index list using the requested stride
# frame_indices = list(range(start_idx, end_idx + 1, frame_step))
# if not frame_indices:
#     raise ValueError("No frames selected. Check start_idx/end_idx/frame_step.")

# # Output location (next to the input NetCDF)
# video_dir = model_nc.parent
# video_dir.mkdir(parents=True, exist_ok=True)
# mp4_path = video_dir / f"{file_tag}_movie_step{frame_step}_fps{fps}.mp4"
# gif_path = video_dir / f"{file_tag}_movie_step{frame_step}_fps{fps}.gif"

# # Create figure and first frame
# fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=200)
# first_idx = frame_indices[0]
# first_t = time_str_all.isel(time=first_idx).item()
# quad = uds[var_name].isel(time=first_idx).ugrid.plot(
#     ax=ax,
#     cmap=cmap,
#     vmin=vmin,
#     vmax=vmax,
#     add_colorbar=True,
#     cbar_kwargs={"label": label},
# )

# cbar = quad.colorbar
# cbar.set_label(label, fontsize=9)
# cbar.ax.tick_params(labelsize=8)

# ax.set_xlabel("Longitude (°)", fontsize=9)
# ax.set_ylabel("Latitude (°)", fontsize=9)
# ax.set_aspect("equal")
# ax.set_facecolor("whitesmoke")
# ax.tick_params(labelsize=8)
# ax.set_title(f"Date: {first_t}", fontsize=9)


# def update(frame_num):
#     idx = frame_indices[frame_num]
#     t_str = time_str_all.isel(time=idx).item()

#     # Remove previous mesh but keep colorbar/axes styling intact
#     if ax.collections:
#         ax.collections[0].remove()

#     uds[var_name].isel(time=idx).ugrid.plot(
#         ax=ax,
#         cmap=cmap,
#         vmin=vmin,
#         vmax=vmax,
#         add_colorbar=False,
#     )
#     # Re-apply labels and styling because ugrid.plot can reset axis labels to x/y.
#     ax.set_xlabel("Longitude (°)", fontsize=9)
#     ax.set_ylabel("Latitude (°)", fontsize=9)
#     ax.set_aspect("equal")
#     ax.set_facecolor("whitesmoke")
#     ax.tick_params(labelsize=8)
#     ax.set_title(f"Date: {t_str}", fontsize=9)
#     return ax.collections


# anim = FuncAnimation(
#     fig,
#     update,
#     frames=len(frame_indices),
#     interval=1000 / fps,
#     blit=False,
# )

# if ffmpeg_ready:
#     writer = FFMpegWriter(fps=fps, bitrate=bitrate)
#     anim.save(mp4_path, writer=writer)
#     saved_path = mp4_path
# else:
#     # Simple fallback when ffmpeg is unavailable.
#     writer = PillowWriter(fps=fps)
#     anim.save(gif_path, writer=writer)
#     saved_path = gif_path
#     print("ffmpeg not found. Saved GIF instead of MP4.")
#     print("Set INSTALL_VIDEO_DEPS = True in this script, run once, then rerun for MP4.")

# plt.close(fig)

# print(f"Saved {movie_var} movie: {saved_path}")
# print(f"Frames: {len(frame_indices)} | Step: {frame_step} timestep(s) | FPS: {fps}")

