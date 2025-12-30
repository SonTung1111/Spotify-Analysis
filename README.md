# Spotify Song Analysis — Business Analysis Capstone Project

Capstone project for a Business Analysis class that demonstrates end-to-end data collection, exploration, preprocessing, hypothesis testing, and basic machine learning on multiple Spotify track datasets. This repository collects public Spotify exports, preprocessing utilities, and notebooks used to support business insights and model-driven recommendations.

## Repository structure

- `fetch_datasets_metadata.py` — helper script to collect or refresh dataset metadata.
- `requirements.txt` — Python dependencies used by notebooks and scripts.
- `notebooks/` — Jupyter notebooks for exploratory analysis, preprocessing, and modeling:
	- `spotify_metadata.ipynb` — dataset exploration and metadata mapping
	- `preprocessing.ipynb` — data cleaning and feature engineering
	- `hypothesis_testing.ipynb` — statistical tests and comparisons
	- `machine_learning_modeling.ipynb` — training and evaluating models
- `src/` — small utilities and training helper code:
	- `trainer.py` — training loop / model wrapper used in experiments
	- `utils.py` — dataset load and preprocessing along with common utilities

## Datasets

The `datasets/` folder must contain the raw and cleaned datasets used by the notebooks. There are two ways to prepare the `datasets/` folder:

- Option A — Use the provided archive (recommended for convenience):
	- Extract the archive and place its contents under a `datasets/` directory at the repository root (create the `datasets/` folder first if it does not exist).
	- After extraction the `datasets/` folder should include the dataset subfolders listed below.

- Option B — Reproduce / re-collect the datasets (reproducible):
	1. Run `python fetch_datasets_metadata.py` to build the initial metadata CSV. This will create `datasets/spotify_datasets_metadata.csv`.
	2. Open and run the `notebooks/spotify_metadata.ipynb` notebook to clean and map metadata. That notebook produces `datasets/spotify_main_datasets_metadata.csv`.
	3. Use the links listed in `datasets/spotify_main_datasets_metadata.csv` to download each dataset (follow the source URLs). Unzip and place the downloaded dataset folders and CSVs under `datasets/`.

Expected `datasets/` structure (examples of folder names you should have after extraction or download):
```text
datasets/
├─ 30000-spotify-songs/
│  ├─ readme.md
│  └─ spotify_songs.csv
├─ spotify-1million-tracks/
│  ├─ spotify_data.csv
├─ spotify-dataset-2023/
│  └─ ...
├─ spotify-tracks-genre-dataset/
│  └─ train.csv
├─ spotify_datasets_metadata.csv
└─ spotify_main_datasets_metadata.csv
```

If you need to update or refresh metadata, run:

```bash
python fetch_datasets_metadata.py
```

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Open the notebooks in `notebooks/` to explore the data and reproduce analyses. Launch Jupyter Lab/Notebook:

```bash
jupyter lab
```

3. Use `src/trainer.py` and `notebooks/machine_learning_modeling.ipynb` as the starting point for model experiments.

## Execution flow

Follow this recommended order to prepare data and reproduce analyses:

1. `fetch_datasets_metadata.py` — collect initial dataset metadata (creates `datasets/spotify_datasets_metadata.csv`).
2. `notebooks/spotify_metadata.ipynb` — clean and map metadata (produces `datasets/spotify_main_datasets_metadata.csv`).
3. `notebooks/preprocessing.ipynb` — clean raw datasets and produce `datasets/spotify_tracks_preprocessed.csv`.
4. `notebooks/hypothesis_testing.ipynb` — run statistical tests and comparisons.
5. `notebooks/machine_learning_modeling.ipynb` — train and evaluate models using the preprocessed data.

**Team / Author**
- Course: Business Analytics
- Author(s): Bui Hai Nam 11385021M; Do Ha My 11385031M; Bui Huu Son Tung 11385062M 
- Instructor: Dr. Ilia Tetin
- Date: 30/12/2025
