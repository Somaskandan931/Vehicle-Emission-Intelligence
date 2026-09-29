# Vehicle Emission Intelligence

### Deep Learning for Real-World Vehicle Emission Prediction and Intelligent Emission Assessment

**Vehicle Emission Intelligence (VEI)** is an end-to-end deep learning system designed to predict vehicle exhaust emissions from real-world driving data and transform those predictions into interpretable emission intelligence.

The project models the temporal relationship between **vehicle operating conditions, environmental conditions, and pollutant emissions** using recurrent neural networks. It compares multiple LSTM architectures and extends emission prediction into higher-level indicators such as **Vehicle Exhaust Emission Intelligence (VEEI)** and **Driver/Vehicle Emission Health Score (DEHS)**.

The goal is not simply to predict the next emission value, but to build a complete pipeline that connects:

> **Driving Behaviour → Vehicle State → Emission Prediction → Emission Intelligence → Interpretable Assessment**

---

## Overview

Vehicle emissions are dynamic.

Acceleration, engine speed, engine load, fuel consumption, ambient conditions, and other driving variables continuously influence pollutant production. Because of this temporal behaviour, treating every emission measurement as an independent observation can miss important patterns in the data.

Vehicle Emission Intelligence approaches the problem as a **multivariate time-series forecasting task**.

The system learns from sequences of vehicle and environmental measurements and predicts multiple pollutants simultaneously:

* **Carbon Monoxide (CO)**
* **Carbon Dioxide (CO₂)**
* **Nitrogen Oxides (NOx)**

The predicted emissions can then be transformed into higher-level emission intelligence through the VEEI and DEHS layers.

---

## The Problem

Traditional vehicle-emission analysis often focuses on measuring emissions after they occur.

This project explores a different approach:

> **Can vehicle operating conditions be used to predict upcoming emissions and provide an interpretable assessment of emission behaviour?**

The system therefore addresses three connected problems:

### 1. Emission Prediction

Predict future pollutant emission rates from recent vehicle and environmental measurements.

### 2. Temporal Modelling

Capture relationships across consecutive observations rather than treating measurements independently.

### 3. Emission Intelligence

Convert predicted pollutant levels into interpretable vehicle-level emission indicators.

---

## How the System Works

```text
┌──────────────────────────────────────────┐
│          Real-World Driving Data         │
│                                          │
│ Speed • RPM • Load • Fuel • Temperature │
│ Ambient Conditions • Emission Sensors   │
└─────────────────────┬────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐
│        Data Processing & Cleaning        │
│                                          │
│ Timestamp Alignment                      │
│ Resampling • Cleaning • Normalisation   │
│ Trip / Vehicle Separation               │
└─────────────────────┬────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐
│          Time-Series Sequences           │
│                                          │
│ Historical vehicle states → Future state│
└─────────────────────┬────────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
      Uni-LSTM    Bi-LSTM    Hybrid LSTM
          │           │           │
          └───────────┼───────────┘
                      ▼
┌──────────────────────────────────────────┐
│         Multi-Pollutant Prediction       │
│                                          │
│              CO • CO₂ • NOx             │
└─────────────────────┬────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐
│          Emission Intelligence           │
│                                          │
│        VEEI → DEHS → Vehicle Insight    │
└─────────────────────┬────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐
│        Interactive Vehicle Dashboard     │
└──────────────────────────────────────────┘
```

---

# Core Machine Learning Approach

## 1. Uni-LSTM

The Uni-LSTM model processes the driving sequence chronologically.

```text
Past Vehicle States
       │
       ▼
    LSTM
       │
       ▼
Emission Prediction
```

It provides a straightforward temporal baseline for learning relationships between previous vehicle states and future emissions.

---

## 2. Bi-LSTM

The Bi-LSTM architecture processes the input sequence in both directions.

```text
             Input Sequence
             /            \
            ▼              ▼
     Forward LSTM    Backward LSTM
            \              /
             └──────┬───────┘
                    ▼
             Emission Output
```

This allows the model to construct a richer representation of the temporal information contained within each input window.

---

## 3. Hybrid LSTM

The Hybrid LSTM combines Uni-LSTM and Bi-LSTM representations.

```text
                 Input Sequence
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          Uni-LSTM            Bi-LSTM
             │                   │
             │              Projection
             │                   │
             └─────────┬─────────┘
                       ▼
                Feature Fusion
                       │
                       ▼
                 Dense Layer
                       │
                       ▼
                CO / CO₂ / NOx
```

The fused representation is defined as:

```text
H = λ · H_Bi + (1 − λ) · H_Uni
```

where `λ` controls the contribution of the bidirectional representation.

The fusion parameter is selected using validation data rather than the final test set.

---

# Real-World Data

The system is designed around real-world vehicle emissions measurements collected using **Portable Emissions Measurement Systems (PEMS)**.

After preprocessing, the working dataset contains approximately:

* **413,971 observations**
* **68 trips**
* **3 vehicle categories**
* **1 Hz temporal resolution**
* Vehicle operating measurements
* Environmental measurements
* CO measurements
* CO₂ measurements
* NOx measurements

The dataset provides the combination of vehicle state and emission information required to study the relationship between driving conditions and pollutant production.

## Dataset

This project uses the **Real Driving Emissions Data: University of Pretoria (Version 5)** dataset published on Mendeley Data by **Johan W. Joubert**.

The dataset contains real-world driving emissions collected using a **Portable Emissions Measurement System (PEMS)** across multiple vehicle types, including the Isuzu FTR850 AMT, Ford Figo 1.5, and Toyota Etios 1.5. Version 5 also includes additional test ensembles recorded at 5 Hz, while emissions are reported per second.

**Dataset:** [Real driving emissions data: University of Pretoria — Version 5](https://data.mendeley.com/datasets/y9pjtt5ngc/5?utm_source=chatgpt.com)

**DOI:** `10.17632/y9pjtt5ngc.5`

**License:** CC BY 4.0

### Dataset Citation

> Joubert, J. W. (2023). *Real driving emissions data: University of Pretoria* (Version 5). Mendeley Data. https://doi.org/10.17632/y9pjtt5ngc.5

The experimental design and PEMS data collection methodology are described in:

> Joubert, J. W., & Gräbe, R. J. (2022). *Real driving emissions: Isuzu FTR850 AMT*. Data in Brief, 41, 107975. https://doi.org/10.1016/j.dib.2022.107975

The accompanying publication describes the collection of real-world emissions and vehicle diagnostics using a PEMS unit and discusses the use of the data for developing predictive models of instantaneous pollutant concentrations.

---

# What the Model Learns

The system uses vehicle and environmental information such as:

### Vehicle Behaviour

* Vehicle speed
* Engine RPM
* Engine load
* Throttle position
* Fuel flow
* Air-fuel related measurements
* Engine temperature

### Environmental Conditions

* Ambient temperature
* Ambient pressure
* Other available environmental measurements

### Emission Targets

* CO
* CO₂
* NOx

These measurements are converted into temporal sequences so that the model can learn how **recent vehicle behaviour influences subsequent emission levels**.

---

# From Prediction to Emission Intelligence

The project goes beyond conventional regression metrics.

After predicting pollutant levels, the system introduces an additional interpretation layer.

## Vehicle Exhaust Emission Intelligence — VEEI

VEEI combines pollutant predictions into a unified emission-intelligence indicator.

Conceptually:

```text
                    Pollutant Predictions
                   /          |           \
                 CO          CO₂          NOx
                  \           |           /
                   \          |          /
                    ▼         ▼         ▼
                   Normalised Emission
                         Levels
                            │
                            ▼
                         VEEI
```

The index incorporates pollutant-specific contributions using configurable pollutant limits and weights.

This allows the system to move from:

> **"How much CO/CO₂/NOx is predicted?"**

towards:

> **"What does the combined predicted emission profile indicate?"**

---

# Driver / Vehicle Emission Health Score — DEHS

The project further transforms VEEI into a more intuitive score:

```text
VEEI
 │
 ▼
Emission Severity
 │
 ▼
DEHS
```

The DEHS layer is designed to provide a compact representation of emission behaviour that can be surfaced through a vehicle-level dashboard.

The scoring parameters are explicitly configurable so that regulatory assumptions and weighting decisions remain separate from the underlying machine-learning model.

---

# Evaluation

The models are evaluated using both conventional regression metrics and a non-learning forecasting baseline.

### Regression Metrics

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **MAPE** — Mean Absolute Percentage Error
* **R²** — Coefficient of Determination
* **NMAE** — Normalised Mean Absolute Error
* **NRMSE** — Normalised Root Mean Squared Error

### Persistence Baseline

The project also evaluates a persistence forecasting strategy:

```text
Predicted(t+1) = Observed(t)
```

This provides a simple reference point for determining whether the deep-learning models provide useful predictive information beyond the assumption that the next emission value will resemble the current one.

---

# Leakage-Aware Time-Series Design

Because vehicle emission data is sequential, the project explicitly considers temporal leakage.

The pipeline ensures that:

* Training, validation, and test observations remain separated.
* Scalers are fitted using training data only.
* Sequence windows do not cross dataset partitions.
* Sequence windows do not cross vehicle boundaries.
* Sequence windows do not cross identified trip boundaries.
* Hybrid-model parameters are selected using validation data.
* Final test data remains isolated for evaluation.

The project also supports **trip-level holdout evaluation**, allowing complete driving trips to be reserved for testing.

This provides a stricter assessment of how the models generalise to previously unseen driving sequences.

---

# Interactive Dashboard

Vehicle Emission Intelligence includes a Flask-based dashboard for visualising model-generated emission intelligence.

The dashboard is designed to turn model outputs into information that is easier to interpret at the vehicle level.

```text
┌──────────────────────────────────────────┐
│           VEHICLE EMISSION INTELLIGENCE  │
├──────────────────────────────────────────┤
│                                          │
│  Vehicle        VEEI          DEHS       │
│  ───────        ────          ────       │
│  Vehicle 01     ...           ...        │
│  Vehicle 02     ...           ...        │
│  Vehicle 03     ...           ...        │
│                                          │
├──────────────────────────────────────────┤
│          Emission Predictions             │
│                                          │
│       CO     CO₂     NOx                 │
│       ──     ───     ───                 │
│                                          │
└──────────────────────────────────────────┘
```

The dashboard separates the **machine-learning pipeline** from the **visualisation layer**, allowing previously generated results to be explored without retraining the models.

---

# Technology Stack

| Area             | Technologies     |
| ---------------- | ---------------- |
| Programming      | Python           |
| Deep Learning    | PyTorch          |
| Machine Learning | Scikit-learn     |
| Data Processing  | Pandas, NumPy    |
| Visualisation    | Matplotlib       |
| Web Dashboard    | Flask            |
| Configuration    | YAML             |
| Testing          | Pytest           |
| Experimentation  | Jupyter Notebook |
| Version Control  | Git              |

---

# Project Architecture

```text
vehicle-emission-intelligence/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── configs/
│   ├── data.yaml
│   ├── uni_lstm.yaml
│   ├── bi_lstm.yaml
│   ├── hybrid_lstm.yaml
│   └── by_trip.yaml
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── emission/
│   └── visualization/
│
├── models/
│   └── checkpoints/
│
├── results/
│   ├── metrics/
│   ├── predictions/
│   ├── tables/
│   └── figures/
│
├── experiments/
│
├── notebooks/
│
├── dashboard/
│
├── reports/
│
└── paper/
```

The architecture separates **data processing, modelling, training, evaluation, emission intelligence, visualisation, and presentation** so that each stage can be independently inspected and reproduced.

---

# Why This Project Matters

Vehicle emissions are influenced by continuously changing driving conditions.

A vehicle does not produce a fixed emission level. Emissions evolve with:

```text
Acceleration
     ↓
Engine State
     ↓
Fuel / Air Behaviour
     ↓
Combustion
     ↓
Pollutant Production
```

Vehicle Emission Intelligence attempts to capture this dynamic relationship using sequential deep learning and then translate model predictions into an interpretable emission assessment layer.

The resulting system brings together:

**Real-world data + Time-series modelling + Multi-pollutant prediction + Emission intelligence + Visual analytics**

into a single research-oriented pipeline.

---

# Project Highlights

* Real-world vehicle emission data
* Multivariate time-series modelling
* Multi-pollutant prediction
* Uni-LSTM architecture
* Bi-LSTM architecture
* Hybrid Uni/Bi-LSTM architecture
* Persistence forecasting baseline
* Leakage-aware temporal evaluation
* Trip-level generalisation analysis
* VEEI emission-intelligence layer
* DEHS vehicle-level scoring
* Interactive Flask dashboard
* Reproducible experiment structure
* Automated testing
* Research-ready results and visualisation pipeline

---

# Research Direction

Vehicle Emission Intelligence is designed as a foundation for further research in:

* Intelligent transportation systems
* Vehicle emission forecasting
* Environmental AI
* Sustainable mobility
* Time-series deep learning
* Driver behaviour analysis
* Vehicle health and efficiency monitoring
* Real-time emission intelligence

Future extensions could incorporate additional pollutants, richer driving-context features, real-time inference, edge deployment, explainable predictions, and larger multi-vehicle datasets.

---

# Project Status

**Vehicle Emission Intelligence is an active research and engineering project.**

The complete pipeline covers:

```text
Data
 ↓
Preprocessing
 ↓
Sequence Construction
 ↓
LSTM Modelling
 ↓
Prediction
 ↓
Evaluation
 ↓
Emission Intelligence
 ↓
Dashboard
```

The repository is structured to keep **experimental implementation, model outputs, and research interpretation clearly separated**, allowing the system to evolve as additional experiments and validation are completed.

---

## Author

**Somaskandan Rajagopal**

M.Sc. Computer Science
Sathyabama Institute of Science and Technology

**Areas of Interest:**
Artificial Intelligence · Machine Learning · Deep Learning · Time-Series Modelling · Explainable AI · Intelligent Transportation Systems · Environmental AI
