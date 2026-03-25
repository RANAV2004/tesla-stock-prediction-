import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Tesla Predictor",
    page_icon="📈",
    layout="wide"
)

# ------------------ TITLE ------------------
st.markdown("""
# 🚀 Tesla Stock Price Prediction
### AI-powered forecasting dashboard
""")

# ------------------ LOAD DATA ------------------
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "../../data/tsla.csv")

df = pd.read_csv(file_path)

# Clean data
df = df[['Close']]
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ------------------ METRICS ------------------
latest_price = df['Close'].iloc[-1]
avg_price = df['Close'].mean()

col1, col2 = st.columns(2)

with col1:
    st.metric("📌 Latest Price", f"${latest_price:.2f}")

with col2:
    st.metric("📊 Average Price", f"${avg_price:.2f}")

# ------------------ HISTORICAL GRAPH ------------------
st.subheader("📊 Historical Tesla Prices")

fig_hist = go.Figure()

fig_hist.add_trace(go.Scatter(
    y=df['Close'],
    mode='lines',
    name='Historical'
))

fig_hist.update_layout(
    template="plotly_dark",
    xaxis_title="Time",
    yaxis_title="Price"
)

st.plotly_chart(fig_hist, use_container_width=True)

# ------------------ SCALING ------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

# ------------------ LOAD MODEL ------------------
model_path = os.path.join(base_dir, "../../models/lstm_model.h5")
model = tf.keras.models.load_model(model_path, compile=False)

# ------------------ FUNCTIONS ------------------
def get_last_sequence(data, time_steps=60):
    return data[-time_steps:]

def predict_future(model, last_sequence, days):
    future = []
    seq = last_sequence.copy()

    for _ in range(days):
        pred = model.predict(seq.reshape(1, 60, 1), verbose=0)
        future.append(pred[0][0])
        seq = np.append(seq[1:], pred)

    return scaler.inverse_transform(np.array(future).reshape(-1,1))

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Controls")

days = st.sidebar.selectbox("Select Prediction Days", [1, 5, 10])
predict_btn = st.sidebar.button("Predict")

# ------------------ PREDICTION ------------------
if predict_btn:
    last_seq = get_last_sequence(scaled_data)
    predictions = predict_future(model, last_seq, days)

    st.subheader("📊 Predicted Prices")

    for i, price in enumerate(predictions):
        st.write(f"Day {i+1}: ${price[0]:.2f}")

    # ------------------ GRAPH ------------------
    pred_values = [price[0] for price in predictions]

    st.subheader("📈 Tesla Forecast")

    fig = go.Figure()

    # Last 100 historical points
    fig.add_trace(go.Scatter(
        y=df['Close'][-100:],
        mode='lines',
        name='Historical'
    ))

    # Future predictions
    future_x = list(range(len(df['Close'][-100:]), len(df['Close'][-100:]) + len(pred_values)))

    fig.add_trace(go.Scatter(
        x=future_x,
        y=pred_values,
        mode='lines+markers',
        name='Predicted'
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)