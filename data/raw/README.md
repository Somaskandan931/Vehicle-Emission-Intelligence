# Raw data

Place the real dataset here as `emissions.csv` (or change `data.raw_file` in `configs/data.yaml`).

Required (names configurable in `configs/data.yaml`):
- `vehicle_id` - vehicle identifier (or set `vehicle_col: null` for a single series)
- `timestamp` - time or ordering column (needed for a *temporal* LSTM)
- target columns: `CO, HC, NOx, CO2, SO2, PM`
- any vehicle / operating features (speed, load, fuel type, engine size, temperature ...)

If the dataset lacks some pollutants, edit `data.target_cols` **and** the paper's claims.
Raw data is git-ignored; document its origin in `data/metadata/dataset_description.md`.

## Dataset in use: University of Pretoria real driving emissions (Mendeley Data y9pjtt5ngc)
1. Unzip the download and put the three files in `data/raw/pretoria/` (or directly in `data/raw/`; both are found):
   `public-etios.csv`, `public-figo.csv`, `public-rrv.csv`.
2. Build the single input file the pipeline reads:
   `python -m src.data.merge_raw`   ->  `data/raw/emissions.csv` + `results/tables/merge_report.csv`
3. `configs/data.yaml` is already set for it (series id, timestamp, targets CO/CO2/NOx, leak-free features).

The dataset measures CO, CO2 and NOx (g/s) only. HC, SO2 and PM are not in it, so the paper must claim
three pollutants. Both `data/raw/pretoria/` and `emissions.csv` are git-ignored.
