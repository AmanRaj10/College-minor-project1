# ☀️ SolarAI – AI-Powered Solar Power Forecasting System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red.svg)
![NASA POWER API](https://img.shields.io/badge/NASA-POWER_API-green.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)
![AWS](https://img.shields.io/badge/AWS-Deployment-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

---

## 📌 Overview

SolarAI is an **AI-powered Solar Power Forecasting System** that predicts solar energy generation using **Deep Learning** and real-time **NASA POWER satellite weather data**.

The project combines **CNN + Bidirectional LSTM + Multi-Head Attention** to forecast solar power output and provides an interactive **Streamlit dashboard** for visualization, cloud analysis, battery optimization, and future energy forecasting.

---

## 🚀 Features

- 🌍 Global Solar Forecasting
- ☀️ NASA POWER API Integration
- 🧠 CNN + BiLSTM + Attention Deep Learning Model
- 📈 Multi-Year Training Support
- 🔮 24–168 Hour Solar Power Forecast
- ☁️ Cloud Cover Heatmap
- 🔋 Battery Optimization
- 📊 Interactive Streamlit Dashboard
- 🐳 Docker Support
- ☁️ AWS Deployment Ready

---

# 🖥️ Dashboard

The dashboard provides

- Solar Power Forecast
- Historical Data Analysis
- Cloud Heatmap
- Battery Optimization
- AI Prediction Metrics
- Interactive World Map
- Multi-Year Training
- Forecast Download (CSV)

---

# 🧠 Deep Learning Architecture

```
NASA POWER API
        │
        ▼
Data Collection
        │
        ▼
Feature Engineering
        │
        ▼
CNN Layers
        │
        ▼
Bidirectional LSTM
        │
        ▼
Multi-Head Attention
        │
        ▼
Dense Layers
        │
        ▼
Solar Power Prediction
```

---

# 📂 Project Structure

```
SolarAI-Power-Forecasting/
│
├── app.py
├── train.py
├── preprocessing.py
├── data_fetch.py
├── model_cnn_lstm.py
├── dataset.csv
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── AWS_DEPLOYMENT.md
├── README.md
│
├── saved_model/
│
├── images/
│
└── LICENSE
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/SolarAI-Power-Forecasting.git

cd SolarAI-Power-Forecasting
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python train.py
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 🌍 Dataset

The project fetches weather and solar irradiance data directly from

**NASA POWER API**

Collected parameters include

- Global Horizontal Irradiance (GHI)
- Temperature
- Wind Speed
- Relative Humidity
- Pressure
- Cloud Cover

---

# 📊 Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-Learn
- Docker
- AWS
- NASA POWER API

---

# 📈 Model Features

✔ CNN Feature Extraction

✔ Bidirectional LSTM

✔ Multi-Head Attention

✔ Time-Series Forecasting

✔ Multi-Year Training

✔ Feature Engineering

✔ Battery Optimization

---

# 🔮 Forecasting

The model predicts

- Next 24 Hours
- Next 48 Hours
- Next 72 Hours
- Next 168 Hours (7 Days)

---

# ☁️ Cloud Analysis

The application visualizes

- Monthly Cloud Cover
- Hourly Cloud Distribution
- Solar Availability
- Peak Solar Hours

---

# 🔋 Battery Optimization

The dashboard provides

- Battery Capacity
- Charging Schedule
- Grid Buy/Sell Price
- Revenue Estimation

---

# 🐳 Docker

Build Docker Image

```bash
docker build -t solarai .
```

Run Container

```bash
docker run -p 8501:8501 solarai
```

---

# ☁️ AWS Deployment

Supports

- Amazon ECS
- Amazon ECR
- Amazon S3
- AWS Fargate

Deployment guide available in

```
AWS_DEPLOYMENT.md
```

---

# 📷 Screenshots

Add your screenshots inside

```
images/
```

Example

```
images/dashboard.png

images/forecast.png

images/heatmap.png
```

---

# 📌 Future Improvements

- Live Weather API
- Mobile Application
- Solar Panel Fault Detection
- Real-Time Notifications
- Energy Cost Optimization
- IoT Integration

---

# 👨‍💻 Author

**K. Dinesh Raj Pal Reddy**

B.Tech CSE (AI & ML)

GitHub: https://github.com/kommareddydinesh

---

# ⭐ Support

If you found this project useful,

⭐ Star this repository

🍴 Fork it

📢 Share it

---

# 📄 License

This project is licensed under the MIT License.
