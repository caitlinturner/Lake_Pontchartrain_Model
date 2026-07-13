# Lake Pontchartrain Hydrodynamic Modeling Workflow

This repo builds and evaluates the **Delft3D-FM** hydrodynamic model of
**Lake Pontchartrain, Louisiana**. It pulls the model forcing data, 
writes the boundary files, validates the model, reviews 2d results and makes movies,
and connects the model to `dorado` particle tracking.

The model itself runs **outside** this repo in the Delft3D-FM software. 
Everything *around* the model run lives here.

## Workflow

1. **SETUP**: build the model's input files (boundary forcing + spillway flow).
2. **Run Delft3D-FM**: done externally in the Delft3D-FM.
3. **REVIEW**: validate, visualize 
4. **DORADO**: run particle exposure-time analysis on the model results.



```
                 ┌─────────────────────────────┐
 USACE stage /   │ SETUP-01a  (BCS discharge)  │
 opening records │  → outputs/bcs/…discharge.csv│
                 └──────────────┬──────────────┘
                                │  
 NOAA + USGS +   ┌──────────────▼──────────────┐
 Open-Meteo data │ SETUP-01  (all forcing)      │
                 │  → outputs/dfm_forcings/<yr>/ │
                 └──────────────┬──────────────┘
                                │  copy .bc / .tim into model folder
                 ┌──────────────▼──────────────┐
                 │  Delft3D-FM  (run externally)│
                 │  → FlowFM_his.nc, FlowFM_map.nc
                 └──────┬──────────┬─────────┬──┘
                        │ (his)    │ (map)   │ (map)
              ┌─────────▼──┐ ┌─────▼──────┐  │
              │ REVIEW-01  │ │ REVIEW-02  │  │
              │ validate   │ │ 2D maps /  │  │
              │ at sites   │ │ movie/share│  │
              └────────────┘ └────────────┘  │
             ┌──────────────────────────────▼───────────────────────┐
             │ DORADO: particle exposure time (from model results) │
             │   DORADO-01 → DORADO-02   whole-system, then spatial │
             │   DORADO-03a → DORADO-03 → DORADO-04   spillway path │
             │   DORADO-05 → DORADO-04   multisource → from_source  │
             └──────────────────────────────────────────────────────┘
```


## Step-by-Step Workflow

### Stage 1: SETUP: build the model inputs

**1a. Spillway flow first.** Run
`notebooks/SETUP-01a-bcs_diversion_leakage_updated.ipynb`. This figures out how
much water goes through the **Bonnet Carré Spillway (BCS)** on any given day, in
two parts:
- **Leakage** when the spillway is *closed* (water still seeps through the gaps
  between the timber "needles"), estimated from the river stage.
- **Operational discharge** when the spillway is *open*, taken straight from the
  USACE records (these always win over the leakage estimate).

It saves a discharge CSV to `outputs/bcs/`, which the next notebook reads.

You only need to do this if you are running July 1, 2026 or after.

**1b. All the other forcing.** Run
`notebooks/SETUP-01-create_hydromet_model_inputs.ipynb`. This pulls everything
the model needs for each year and writes it out in Delft3D-FM format:
- **Tributary discharge** (USGS gauges, scaled per river)
- **Water level** at the open ocean boundary (NOAA tide station, with backups)
- **Wind + air pressure** (Open-Meteo ERA5)
- Optional **air / water / river temperature**, **salinity**, and a **composite
  meteo** file (humidity, air temp, cloud, solar) for the heat model
- The **spillway flow** from step 1a

Output goes to `outputs/dfm_forcings/<year>/` as `.bc` and `.tim` files, plus
QA Excel tables in `outputs/tables/` and a station-location map.

### Stage 2: Run Delft3D-FM (external)

1. Copy the new `.bc` and `.tim` files from `outputs/dfm_forcings/<year>/` into
   your Delft3D-FM model folder (the same folder as `FlowFM.mdu`). They keep the
   same names, so they overwrite the old versions.
2. You usually **don't** need to edit `FlowFM.ext` / `FlowFM_bnd.ext`: those
   point at file *names*, and the names still match. (The one exception is if you
   are turning on composite temperature; then add `_solarradiation` to the end of
   `QUANTITY = humidity_airtemperature_cloudiness` in `FlowFM.ext`.)
3. Run the model. It produces `FlowFM_his.nc` (values at station points) and
   `FlowFM_map.nc` (the full 2D field over time).
4. In Delft3D, you will need to change the timeframe in 4 places. You need to change the
   the model timing in the general tab (Start Time and End Time), and in the outputs tab 
   update the start time and end time for all three outputs. 

### Stage 3: REVIEW + DORADO: analyze the results

**Validate** with `REVIEW-01`, **look at the 2D fields** with `REVIEW-02`, then
run the **DORADO** particle notebooks (see the guide below for the order).

## Notebook Guide

Notebooks are named by stage and are meant to run roughly top to bottom.

### SETUP
- **`SETUP-01-create_hydromet_model_inputs.ipynb`**
  - *Reads:* live NOAA/USGS/Open-Meteo data + the BCS discharge CSV from
    `SETUP-01a`. *Writes:* Delft3D-FM boundary files in
    `outputs/dfm_forcings/<year>/`, plus QA Excel tables and a location map.
  - This is the main input-builder. Toggles at the top let you turn temperature
    and salinity forcing on or off.
- **`SETUP-01a-bcs_diversion_leakage_updated.ipynb`**
  - *Reads:* USACE stage + opening records from `data/external/`. *Writes:* the
    BCS leakage/discharge CSV in `outputs/bcs/`.
  - To update it: add new stage rows to the USACE text file, and if the spillway
    opened, add a column to the opening spreadsheet: then Run All.

### REVIEW
- **`REVIEW-01-validation_at_sites.ipynb`**
  - *Reads:* `FlowFM_his.nc` + live NOAA/USGS observations. Compares modeled
    **water level, salinity, and temperature** at chosen stations using
    Willmott's *d*, Pearson *r*, and RMSE, and prints a **bias hint** telling you
    what datum offset would zero out the mean bias.
  - Pick which variables to check with the `ACTIVE_TARGETS` toggle at the top.
- **`REVIEW-02-model_results_spatial.ipynb`**
  - *Reads:* `FlowFM_map.nc`. Makes 2D map figures (velocity, water level, temp,
    salinity), and optionally a movie and a smaller NetCDF you can share with
    collaborators.

### DORADO (particle exposure time)
Run these in roughly this order; later ones reuse `walk_data` saved by earlier
ones.
- **`DORADO-01-unstructured_grid_exposuretime.ipynb`**
  - Turns the unstructured Delft3D-FM output into a regular grid, seeds particles
    randomly across the lake, runs the particle routing,
    and computes **whole-system exposure time**. Saves `walk_data` and a rasterized
    `.npz` of model variables. The `RANDOM_SEED` makes the particle sample
    reproduciblee.
- **`DORADO-02-spatial_exposuretime.ipynb`**
  - *Reads:* the `walk_data` from DORADO-01. Maps exposure time in 2D.
  	We get system-wide and localized exposure time from this. This notebook
	matches the one in the dorado package. 
- **`DORADO-03a-create_spillway_particles_selected_years.ipynb`**
  - Turns the BCS discharge CSV into a **daily particle-release schedule** (one
    CSV per selected year) using smoothing and a scale factor.
- **`DORADO-03-spillway_release_workflow.ipynb`**
  - *Reads:* the spillway particle CSV from DORADO-03a. Releases particles from a
    fixed spillway line on that daily schedule and tracks them to get exposure
    time for a spillway event. Saves `walk_data`.
- **`DORADO-04-from_source_exposuretime.ipynb`**
  - *Reads:* `walk_data` from a source release: either the spillway run above or
    the multisource run below. Computes **exposure time** of particles origionating 
	from a source. This is similar to nourishment for that source using `from_source_exposure()`.
- **`DORADO-05-multisource_release_workflow.ipynb`**
  - The generalized version of DORADO-03: release particles from **any number of
    named sources** (new orleans canals, spillway, the causeway, tributaries, etc.), each with its own
    location and schedule. Saves the combined `walk_data`, which DORADO-04 reads.

> Heads up: both the spillway path (`DORADO-03a → 03 → 04`) and the multisource
> path (`DORADO-05 → 04`) end at **DORADO-04**, which measures exposure time from
> the saved `walk_data`.

## Where the Data Comes From

- **NOAA CO-OPS** (tidesandcurrents.noaa.gov): water level, wind, air pressure,
  and water temperature at coastal stations.
- **USGS NWIS** (waterdata.usgs.gov): tributary discharge, gauge height,
  salinity, and water temperature.
- **Open-Meteo ERA5** (open-meteo.com): air temperature, humidity, cloud cover,
  solar radiation, wind, and pressure at a grid point.
- **USACE** (rivergages.mvr.usace.army.mil): Bonnet Carré river stage and the
  recorded spillway opening events.

NOAA, USGS, and Open-Meteo are downloaded automatically when the notebooks run.
The USACE files are downloaded by hand and dropped into `data/external/`.

## Core Source Modules

The notebooks stay focused on the workflow; the reusable code lives in `src/`.

- **`src/hydromet_functions.py`**: builds the forcing series and
  writes Delft3D-FM boundary files (discharge/water level `.bc`, wind and
  pressure `.tim`, temperature/salinity `.bc`, and the composite meteo `.tim`).
- **`src/validation_functions.py`**: downloads observations (NOAA CO-OPS, USGS
  NWIS), preprocesses them, and runs the `validate_variable()` model-vs-obs
  comparison used by `REVIEW-01`.
- **`src/figures.py`**: plotting helpers (tributary discharge, meteorology,
  ERA5 panels, and the multi-variable summary grid).
- **`src/dorado_utils.py`**: shared DORADO helpers: `from_source_exposure()`,
  `build_daily_series()`, `resolve_seed_cells()`, and `daily_to_steps()`.

## Environment Setup (Windows)

Use conda with the provided environment file.

```bat
cd path\to\Lake_Pontchartrain_Model
conda env create -f environment.yml
conda activate Lake_Pontchartrain_Model
```

Register the environment as a Jupyter kernel so you can select the enivronment
in the notebooks (basically you can open the VS Code app and it will run).

```bat
python -m ipykernel install --user --name Lake_Pontchartrain_Model --display-name "Python (Lake_Pontchartrain_Model)"
```

Launch the notebooks from the `notebooks/` folder. They set
`PROJECT_ROOT = Path.cwd().parent`, which needs to resolve to the repository root
so that `src/` can be imported.

## Things to Update Before You Run

- **Model output paths.** `REVIEW-01`, `REVIEW-02`, `DORADO-01`, `DORADO-02`, and
  `DORADO-03` each have a hardcoded path near the top pointing at a local
  Delft3D-FM output folder. Change it to match your machine before running if not on lab computer.
- **Years / run identifiers.** Most notebooks have a `model_run`, `YEARS`, or
  `SELECTED_YEARS` setting near the top: set these to the run you are working on.
- **Network hiccups.** The live NOAA/USGS/Open-Meteo pulls occasionally time out.
  If a cell fails with a connection error, just re-run that cell. 

## Directory Structure

```
data/external/            USACE stage + opening files (hand-downloaded)
notebooks/                the workflow notebooks
notebooks/data/           small helper files (grid cell types, boundaries)
outputs/bcs/              BCS leakage/discharge CSV (from SETUP-01a)
outputs/dfm_forcings/<year>/   Delft3D-FM boundary + meteo files (from SETUP-01)
outputs/tables/           QA Excel tables
outputs/figures/          figures and animations
outputs/dorado/           DORADO walk_data and exposure-time products
src/                      reusable Python modules
```

## Notes

- This workflow makes some large binary/text outputs (NetCDF, particle
  `walk_data`, animations) and may take a while to run. 





