
# Electricity Load Forecasting & Anomaly Detection Dashboard

An end-to-end deep learning pipeline and interactive Streamlit application for multi-client 24-hour electricity load forecasting and statistical anomaly detection. Built on the **UCI ElectricityLoadDiagrams20112014** dataset, this project evaluates deep sequence models against domain-standard temporal baselines.

--- 
##  Dashboard Preview

![Dashboard Preview](Electricity_Load_Forecast_&_Anomaly_Dashboard_page.jpg)

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
├── app.py                   # Main Streamlit dashboard script
├── Energy_forcasting.ipynb  # Data preprocessing, training, and evaluation pipeline
├── requirements.txt         # Python package dependencies
├── .gitignore               # Excludes raw data (>100MB) and log files
├── LICENSE                  # MIT License
└── artifacts/               # Serialized model and evaluation artifacts
    ├── model.keras          # Trained deep learning model
    ├── scaler.pkl           # MinMaxScaler object
    ├── client_cols.pkl      # List of client IDs
    ├── threshold.pkl        # Anomaly threshold boundary
    ├── metrics.csv          # Pre-computed client evaluation metrics
    └── sample_df.pkl        # Lightweight sample fallback data

```

---

## Evaluation Metrics & Baselines

The model performance is benchmarked directly against domain baselines to verify prediction quality:

| Metric | Description |
| --- | --- |
| **`model_mae`** | Mean Absolute Error (kW) achieved by the trained sequence model |
| **`yesterday_mae`** | Baseline MAE using load values from the exact same hour yesterday ($t-24$) |
| **`last_week_mae`** | Baseline MAE using load values from the exact same hour last week ($t-168$) |
| **`model_smape`** | Symmetric Mean Absolute Percentage Error of the forecast horizon |

---

##  Quick Start & Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/Adibshaikh2209/energy-load-forecasting-anomaly-detection.git](https://github.com/Adibshaikh2209/energy-load-forecasting-anomaly-detection.git)
cd energy-load-forecasting-anomaly-detection

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Launch the Streamlit Dashboard

```bash
streamlit run app.py

```

Your default browser will automatically open `http://localhost:8501`.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

Would you like help setting up the `git commit` commands after saving this file?

```
