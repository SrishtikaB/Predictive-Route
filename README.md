# PredictiveRoute

### AI-Based Network Congestion Prediction and Prevention Using Machine Learning and Rule-Based Traffic Routing

PredictiveRoute is an academic project that explores network congestion prediction and congestion-aware traffic routing using machine learning and rule-based decision-making.

## Overview

Network congestion can increase latency, cause packet loss, and reduce overall network performance. PredictiveRoute explores how machine learning and routing decisions based on network conditions can help address these challenges in a simulated environment.

## Features

- **Machine Learning:** Uses a Random Forest classifier to predict network congestion from network traffic and performance features.
- **Traffic Routing:** Simulates routing decisions based on network conditions.
- **Network Topology:** Provides a visual representation of network nodes and connections.
- **Performance Analysis:** Displays simulation performance metrics.
- **Interactive Dashboard:** Uses Streamlit to present network information and simulation results.

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- NetworkX
- Matplotlib

## Project Structure

```text
Predictive-Route/
├── dashboard.py
├── dynamic_network.py
├── main.py
├── main_original.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/SrishtikaB/Predictive-Route.git
cd Predictive-Route
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run dashboard.py
```

## Project Status

Academic prototype. Network behavior and performance metrics depend on the simulation configuration and generated data.

## Author

**Srishtika Bhatnagar**
