import os
import shutil
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

st.set_page_config(page_title="Electricity Load Forecast & Anomaly Dashboard", layout="wide")

# Custom CSS for compact zero-scroll fit
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
        h1 { margin-bottom: 0rem; font-size: 1.8rem; }
        h3 { font-size: 1.2rem; margin-top: 0.5rem; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    possible_paths = [
        'artifacts',
        './artifacts',
        '/content/drive/MyDrive/energydiagram/artifacts',
        '/content/artifacts'
    ]
    artifacts_dir = next((p for p in possible_paths if os.path.exists(p)), None)

    if artifacts_dir is None:
        st.error("Artifacts directory not found.")
        st.stop()

    model_path = os.path.join(artifacts_dir, 'model.keras')

    # Self-healing check: convert extracted directory back to zip archive if necessary
    if os.path.exists(model_path) and os.path.isdir(model_path):
        temp_zip = os.path.join(artifacts_dir, 'model_temp')
        shutil.make_archive(temp_zip, 'zip', model_path)
        shutil.rmtree(model_path)
        shutil.move(f"{temp_zip}.zip", model_path)

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.pkl'))
    client_cols = joblib.load(os.path.join(artifacts_dir, 'client_cols.pkl'))
    threshold = joblib.load(os.path.join(artifacts_dir, 'threshold.pkl'))
    df_metrics = pd.read_csv(os.path.join(artifacts_dir, 'metrics.csv'))
    sample_df = joblib.load(os.path.join(artifacts_dir, 'sample_df.pkl'))

    X_test_path = os.path.join(artifacts_dir, 'X_test.npy')
    y_test_path = os.path.join(artifacts_dir, 'y_test.npy')

    if os.path.exists(X_test_path) and os.path.exists(y_test_path):
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
    else:
        X_test, y_test = None, None

    return model, scaler, client_cols, threshold, df_metrics, sample_df, X_test, y_test

model, scaler, client_cols, threshold, df_metrics, sample_df, X_test, y_test = load_artifacts()

st.title("⚡ Electricity Load Forecasting & Anomaly Detection Dashboard")
st.caption("UCI ElectricityLoadDiagrams20112014 Dataset (24-Hour Horizon)")

# Sidebar Controls
st.sidebar.header("Configuration")
selected_client = st.sidebar.selectbox("Select Client", client_cols)

if X_test is not None and y_test is not None:
    sample_idx = st.sidebar.slider("Select Forecast Window (Test Sequence Index)", 0, len(X_test) - 1, 100)

    c_idx = client_cols.index(selected_client)
    num_clients = len(client_cols)

    single_X = X_test[sample_idx:sample_idx+1]
    y_pred_scaled = model.predict(single_X, verbose=0)

    flat_true = y_test[sample_idx].reshape(-1, num_clients)
    flat_pred = y_pred_scaled[0].reshape(-1, num_clients)

    actual_unscaled = scaler.inverse_transform(flat_true)[:, c_idx]
    pred_unscaled = scaler.inverse_transform(flat_pred)[:, c_idx]

    residuals = np.abs(actual_unscaled - pred_unscaled)
    anomalies = residuals > threshold

    hours = np.arange(1, 25)

    # --- TOP ROW: Graph (Left) & Anomaly Metrics (Right) ---
    col_graph, col_anomaly = st.columns([3.2, 1])

    with col_graph:
        st.subheader(f"24-Hour Actual vs Forecast for {selected_client}")
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(hours, actual_unscaled, label='Actual (kW)', marker='o', color='#1f77b4', linewidth=1.5, markersize=4)
        ax.plot(hours, pred_unscaled, label='Forecast (kW)', linestyle='--', marker='s', color='#ff7f0e', linewidth=1.5, markersize=4)

        if np.any(anomalies):
            ax.scatter(hours[anomalies], actual_unscaled[anomalies], color='red', s=80, label='Anomaly Flagged', zorder=5)

        ax.set_xlabel("Hour Ahead", fontsize=9)
        ax.set_ylabel("Electricity Load (kW)", fontsize=9)
        ax.set_xticks(hours)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_anomaly:
        st.subheader("Anomaly Detection")
        st.metric(label="Anomaly Threshold (Val Mean + 3σ)", value=f"{threshold:.2f} kW")
        st.divider()
        st.metric(label="Flagged Anomalies in Window", value=int(np.sum(anomalies)))

    # --- BOTTOM ROW: Compact Metrics Table ---
    st.subheader("Client Evaluation Metrics")
    client_row = df_metrics[df_metrics['client'] == selected_client]
    avg_row = df_metrics[df_metrics['client'] == 'AVERAGE']
    display_metrics = pd.concat([client_row, avg_row], ignore_index=True)
    st.dataframe(
        display_metrics[['client', 'model_mae', 'yesterday_mae', 'last_week_mae', 'model_smape']],
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("Running in lightweight cloud preview mode using pre-saved sample dataset.")
    st.line_chart(sample_df[selected_client])
    st.subheader("Baseline Metrics Table")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
