# DeliveryNet AI 🚚

**Intelligent Logistics Route Optimisation & Fleet Simulation Engine**

An AI-powered logistics simulation platform that combines machine learning,
graph algorithms, and fleet management into an interactive Streamlit dashboard.

---

## Features

| Component | Technology |
|---|---|
| City Network Generator | NetworkX directed graph with haversine distances |
| Traffic Prediction Model | Random Forest Regressor (scikit-learn) |
| Route Optimisation | Dijkstra & A* with composite cost function |
| Fleet Management | OOP vehicle simulation with priority assignment |
| Delivery Simulation | Time-step engine with full KPI tracking |
| Dashboard | Streamlit + Plotly enterprise-style UI |

---

## Project Structure

```
deliverynet-ai/
│
├── app.py                    # Streamlit entry point
├── config.py                 # All configuration constants
│
├── core/
│   ├── city_network.py       # NetworkX city graph generator
│   ├── traffic_model.py      # Traffic model bridge
│   ├── route_optimizer.py    # Dijkstra / A* optimisation
│   ├── fleet_manager.py      # Vehicle & order management
│   └── delivery_simulator.py # Time-step simulation engine
│
├── models/
│   └── traffic_predictor.py  # Random Forest traffic predictor
│
├── ui/
│   ├── dashboard.py          # Streamlit page layout
│   ├── charts.py             # Plotly figure factories
│   └── controls.py           # Sidebar widgets
│
├── utils/
│   └── helpers.py            # Shared utility functions
│
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dashboard Walkthrough

1. **Sidebar** – Configure fleet size, number of deliveries, time steps, and route
   optimiser cost weights.
2. **Run Simulation** – Click the button to build the city network, train the ML
   model, and execute the full time-step simulation.
3. **KPI Cards** – View total orders, delivery success rate, fleet utilisation,
   delayed deliveries, and average fuel level.
4. **City Network Map** – Interactive geo-scatter showing warehouses, delivery
   locations, and road edges coloured by congestion severity.
5. **Traffic Heatmap** – Congestion factor by road type with error bars.
6. **Delivery Timeline** – Cumulative completed vs pending deliveries over time.
7. **Fleet & Fuel Chart** – Fleet utilisation percentage and average fuel level.
8. **Fleet & Orders Tables** – Live status of every vehicle and order.
9. **ML Insights** – Model MAE, R², and feature importance rankings.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** – dashboard framework
- **scikit-learn** – Random Forest traffic predictor
- **NetworkX** – city graph and shortest-path algorithms
- **Plotly** – interactive charts
- **Pandas / NumPy** – data manipulation

---

## Deployment

Deploy to [Streamlit Cloud](https://streamlit.io/cloud) in one click:

1. Push this repository to GitHub.
2. Connect to Streamlit Cloud and point to `app.py`.
3. Add `requirements.txt` — Streamlit Cloud installs dependencies automatically.
