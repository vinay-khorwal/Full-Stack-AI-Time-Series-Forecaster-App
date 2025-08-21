# 📈 Full-Stack AI Time Series Forecaster App

A complete full-stack web application that allows users to **upload their own time series data** and generate **future trend forecasts** using a **Transformer-based deep learning model**.

The app features a **React frontend** for an interactive user experience and a **Django backend** to handle data processing and the ML pipeline. It is robust, user-friendly, and capable of handling volatile time series datasets such as stock market or sales data.

<!-- ---

## 🚀 Live Demo Screenshot
*(Example: Forecasting the future "Volume" of ADANIPORTS stock dataset — handling volatile financial data with ease.)* -->

---

## ✨ Key Features

- **Dynamic Time Series Detection**  
  Automatically detects a valid date/time column from uploaded CSV or Excel files.

- **Interactive Column Selection**  
  Lets users choose which numeric metric they want to forecast.

- **Autoregressive Future Forecasting**  
  Trains on **100% of user data** and predicts the next **20% of the timeline**.

- **Robust Data Handling**  
  - Handles datasets of different sizes.  
  - Converts data to correct numeric formats.  
  - Provides clear error messages for small datasets.

- **Logarithmic Transformation for Volatility**  
  Applies log-transform for high-volatility data (e.g., stock volume), ensuring stable predictions.

- **Modern Tech Stack**  
  - **Frontend**: React, Axios, Recharts, CSS3  
  - **Backend**: Django REST Framework, PyTorch, Pandas, NumPy, Scikit-learn

- **Clear Visualizations**  
  Presents forecasts in interactive charts and detailed tables.

---

## 🛠 How It Works: The Two-Step Architecture

### **Step 1: File Analysis**
1. User uploads a file via the React UI.  
2. File sent to Django endpoint: `/api/analyze/`.  
3. Backend detects time column, identifies numeric columns, returns info to frontend.  

### **Step 2: Forecasting**
1. User selects a column to forecast.  
2. React sends request to `/api/predict/`.  
3. Backend runs forecasting pipeline:
   - Trains a Transformer model on 100% of data.  
   - Autoregressively predicts the next 20%.  
4. Frontend renders results in chart + table.  

---

## 🧠 The Machine Learning Pipeline

The **`ml_pipeline.py`** is the heart of the backend ML logic.

1. **Data Loading**  
   - Accepts CSV & Excel.  
   - Detects date column & converts other columns to numeric.  

2. **Feature Engineering**  
   - **Lag Features** (e.g., values 1, 7, 14 days ago).  
   - **Cyclical Features** (date encoded with sine/cosine for seasonality).  

3. **Transformer Model**  
   - Based on attention mechanism.  
   - Learns historical patterns & makes autoregressive predictions.  

4. **Autoregressive Forecasting**  
   - Predicts one step ahead → feeds prediction back → repeats until forecast horizon is reached.  

---

## 📂 Project Structure

```bash
/timeseries_predictor_app
├── /backend
│   ├── /api
│   │   ├── ml_pipeline.py   # Core ML logic
│   │   ├── views.py         # API endpoints
│   │   └── urls.py
│   ├── /predictor_project
│   │   └── settings.py      # Django project config
│   └── manage.py
│
└── /frontend
    ├── /src
    │   └── App.js           # Main React component
    └── package.json
```

---

## ⚙️ Technology Stack

### Backend
- Python  
- Django & Django REST Framework  
- PyTorch (Transformer model)  
- Pandas & NumPy  
- Scikit-learn  

### Frontend
- React.js  
- Axios (API communication)  
- Recharts (interactive charts)  
- CSS3  

---

## 🖥 Setup & Installation

### Prerequisites
- Git  
- Python **3.9+**  
- Node.js **16+**

---

### 🔹 1. Clone the Repository
```bash
git clone <your-repository-url>
cd timeseries_predictor_app
```

### 🔹 2. Backend Setup (Django)
```bash
cd backend

# Create & activate virtual environment
python -m venv venv

# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Install dependencies
pip install django djangorestframework django-cors-headers pandas numpy torch scikit-learn openpyxl
```

### 🔹 3. Frontend Setup (React)
```bash
cd ../frontend

# Install dependencies
npm install
```

---

## ▶️ Running the Application

Run both backend & frontend servers simultaneously.

### 1. Start Backend
```bash
cd backend
python manage.py runserver
```
Runs at: http://localhost:8000

### 2. Start Frontend
```bash
cd frontend
npm start
```
Runs at: http://localhost:3000

---

## 📊 Usage Guide

- Upload a CSV/Excel time series dataset.
- Click "Analyze File".
- Select the target column from the dropdown.
- Click "Generate Forecast".
- View historical + future predictions in chart & table.

---

## 🔮 Future Improvements

- Allow users to select different models (Transformer, LSTM, ARIMA).
- Implement Optuna hyperparameter caching.
- Dockerization for easier deployment.
- Add user authentication to save datasets & forecasts.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.



---

<!--
Do you also want me to **add GitHub-style badges** (for Python, React, License, etc.) at the very top of the README to make it look more professional and open-source friendly?
-->