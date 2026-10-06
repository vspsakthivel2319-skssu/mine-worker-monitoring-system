# ⛏️ Mine Workers Monitoring System

A Python and Streamlit-based web application for monitoring mine workers using health parameters, gas levels, and machine learning.

## Features

- 👷 Monitor all workers at the same time
- 🔎 Search for an individual worker
- 🟢 NORMAL condition
- 🟠 WARNING condition
- 🔴 CRITICAL condition
- Display reasons for Warning and Critical conditions
- Monitor health parameters
- Monitor mine gas levels
- Machine Learning-based condition classification
- CSV and Excel dataset upload
- Interactive dashboard

## Health Parameters

- Heart Rate
- SpO₂
- Temperature
- Blood Pressure
- Pulse
- Respiratory Rate

## Gas Levels

- CH₄ Methane
- CO Carbon Monoxide
- CO₂ Carbon Dioxide
- H₂S Hydrogen Sulfide
- O₂ Oxygen

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- OpenPyXL

## Project Structure

```text
mine_worker_monitor/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
├── assets/
│
└── .streamlit/
    └── config.toml