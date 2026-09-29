# Dataset description  (fill in - the reviewers explicitly asked for this)

| Item | Value |
|---|---|
| Name | Real driving emissions data: University of Pretoria (Joubert & Graebe) |
| Source / URL / publisher | Mendeley Data, dataset y9pjtt5ngc (https://data.mendeley.com/datasets/y9pjtt5ngc); University of Pretoria; companion paper in Data in Brief (Isuzu FTR850 AMT) |
| Licence / citation | |
| Collection method (on-road, dynamometer, remote sensing ...) | On-road PEMS (SEMTECH DS+), urban route in South Africa |
| Collection period | 2021-02-02 to 2022-12-10 (per-vehicle ranges in `results/tables/merge_report.csv`) |
| Number of records | 1,190,091 raw rows (etios 273,493; figo 167,559; rrv 749,039); 413,971 rows after 1 Hz resampling |
| Number of vehicles | 3 vehicles, 68 trips (etios 10, figo 30, rrv 28) |
| Vehicle types / fuel types | Toyota Etios 1.5 and Ford Figo 1.5 (light passenger cars), Isuzu FTR850 AMT road-rail truck (medium heavy vehicle); fuel type is not a column in the files: cite the dataset paper before stating it |
| Sampling interval | 1 Hz (figo, some rrv trips) and 5 Hz (etios, other rrv trips); all trips resampled to 1 Hz by per-second mean |
| Pollutants + units | CO, CO2, NOx in g/s (`CO_mass`, `CO2_mass`, `NOx_mass`). HC, SO2 and PM are not measured |
| Operating conditions | Speed, rpm, throttle, manifold pressure/temperature, coolant temperature, fuel flow/rate, air-fuel ratio, GPS altitude/speed, ambient temperature/humidity/pressure, load (rrv: 0/1500/3000 kg), cold start flag |
| Missing-value handling | see `results/tables/cleaning_report.csv` |

## Processing applied by `src/data/merge_raw.py`
Timestamp parsed (+0200 dropped); 1 Hz per-second mean; one series per trip (`<vehicle>_t<trip>_l<load>`), split at
gaps longer than 5 s; exhaust-side columns removed (target leakage); tiny negative analyser readings clipped to 0.
