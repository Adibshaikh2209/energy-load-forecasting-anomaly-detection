# Electricity Load Forecasting & Anomaly Detection Dashboard

An end-to-end deep learning pipeline and interactive Streamlit application for multi-client 24-hour electricity load forecasting and statistical anomaly detection. Built on the **UCI ElectricityLoadDiagrams20112014** dataset, this project evaluates deep sequence models against domain-standard temporal baselines.

---

## Dashboard Preview

<img src="Electricity Load Forecast & Anomaly Dashboard_page.jpg" width="100%" alt="Dashboard Preview">

---

## Project Overview

Accurate multi-step electricity load forecasting is critical for grid stability, peak-demand management, and anomaly mitigation. This project implements a 24-step ahead multi-client forecasting system coupled with an automated residual-based anomaly detector.

- **Forecasting Horizon**: 24 hours ahead
- **Target Metrics**: Model MAE, Yesterday Baseline MAE ($t-24$), Last Week Baseline MAE ($t-168$), and Model sMAPE
- **Anomaly Thresholding**: Statistical residual boundary calculated as $\text{Mean} + 3\sigma$ from validation residuals
- **User Interface**: Compact, single-screen interactive Streamlit dashboard

---

## Project Structure

```text
energy-load-forecasting-anomaly-detection/
├── app.py                                              # Main Streamlit dashboard script
├── Energy_forcasting.ipynb                             # Data preprocessing, training, and evaluation pipeline
├── Electricity Load Forecast & Anomaly Dashboard_page.jpg  # Dashboard preview image
├── requirements.txt                                    # Python package dependencies
├── .gitignore                                          # Excludes raw data (>100MB) and log files
├── LICENSE                                             # MIT License
└── artifacts/                                          # Serialized model and evaluation artifacts
    ├── model.keras                                     # Trained deep learning model
    ├── scaler.pkl                                      # MinMaxScaler object
    ├── client_cols.pkl                                 # List of client IDs
    ├── threshold.pkl                                   # Anomaly threshold boundary
    ├── metrics.csv                                     # Pre-computed client evaluation metrics
    └── sample_df.pkl                                   # Lightweight sample fallback data
