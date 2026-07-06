# Delft3D-FM 2013 Calibration Check

Post-processing and validation of a calibrated Delft3D-FM model for 2013.
Structured so the same notebook can be re-run for 2014 and 2015 by changing
a single `YEAR` parameter near the top of the notebook.

---

## 1. One-time setup (Windows)

### a. Install Miniforge3

If you don't already have conda/mamba, install Miniforge3 — it ships with
`mamba` and defaults to the conda-forge channel, which is what we want.

- Download the Windows x86_64 installer from https://conda-forge.org/download/
- Install with the recommended settings (install for "Just Me", leave "Add to PATH" unchecked).
- From the Start Menu, open **Miniforge Prompt** (not regular Command Prompt — this prompt has conda activated).

### b. Create the environment

From Miniforge Prompt, `cd` to this project folder:

```bat
cd path\to\Lake_Pontchartrain_Model
conda env create -f environment.yml
```

The first build takes 5–10 minutes (it's downloading ~2 GB of geospatial wheels
via pip after conda installs the base). Go make coffee.

### c. Activate and register the Jupyter kernel

```bat
conda activate Lake_Pontchartrain_Model-dev
python -m ipykernel install --user --name Lake_Pontchartrain_Model-dev --display-name "Python (Lake_Pontchartrain_Model-dev)"
```

The second command makes this env appear as a selectable kernel in any Jupyter
instance on your machine — useful if you're running JupyterLab from a different
base install.

### d. Verify the install

```bat
python -c "import dfm_tools, hydrolib.core, xugrid, xarray; print(dfm_tools.__version__)"
```

Should print a version number (0.45.x or newer) with no errors.

---

## 2. Running Jupyter Notebooks in VSCode

From Miniforge Prompt, with the env activated:

```bat
cd path\to\Lake_Pontchartrain_Model
code . 
```

Open `notebooks/postprocessing_template.ipynb` and confirm the kernel in the top-right
reads **Python (Lake_Pontchartrain_Model-dev)**.

---

## 3. Project layout

```
dfm_2013_project/
├── environment.yml             # conda env spec (this env)
├── README.md                   # you are here
├── notebooks/
│   └── postprocessing_template.ipynb   # the reusable notebook
├── src/                        # .py modules extracted from the notebook
│   └── __init__.py             # as code matures, move reusable functions here
├── data/
│   ├── model_output/           # .nc map/his files from Delft3D-FM runs
│   ├── observations/           # gauge / stage / discharge observation files
│   └── external/               # anything else (shapefiles, DEMs, etc.)
└── outputs/
    ├── figures/                # figures go here, organized by YEAR
    └── tables/                 # stats tables (CSV/XLSX), organized by YEAR
```

**Why this layout:** everything year-specific lives in notebook settings and inside
`outputs/{figures,tables}/<year>/`. The `src/` modules stay reusable.
To run 2014, edit the year and paths in the notebook settings cell, restart the kernel, then run all.

---

## 4. Reusing this for 2014 and 2015

1. Open the notebook settings cell.
2. Change the year setting from 2013 to 2014 (or 2015).
3. Update model and observation paths if needed.
4. In JupyterLab: restart kernel and run all cells.
5. Figures and tables land in `outputs/figures/2014/` and `outputs/tables/2014/`.



