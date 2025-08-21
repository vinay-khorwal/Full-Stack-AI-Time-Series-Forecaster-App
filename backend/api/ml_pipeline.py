# backend/api/ml_pipeline.py

# --- 1. Imports ---
# Standard library imports
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Suppress warnings for a cleaner output in the web server logs
warnings.filterwarnings('ignore')

# --- 2. Global Configuration ---
# These constants define the model's architecture and feature engineering.
# They must match the configuration the model was designed for.
SEQUENCE_LENGTH = 30
LAG_FEATURES = [1, 7, 14]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 3. Data Loading and Feature Engineering Functions ---

def load_and_detect_data(file_path, target_column=None):
    """
    Loads, cleans, and prepares a dataframe from a file path with robust logic
    for handling pre-selected vs. inferred target columns.
    """
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip') if file_path.endswith('.csv') else pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Could not read the file. Ensure it is a valid CSV or Excel file. Error: {e}")

    df.columns = df.columns.str.strip()
    
    # 1. Detect Time Column
    time_col = next((col for col in df if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]) and pd.to_datetime(df[col], errors='coerce').notna().sum() / len(df) > 0.8), None)
    
    if time_col:
        print(f"Detected time series column: '{time_col}'")
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce').dt.normalize()
        df = df.sort_values(by=time_col).reset_index(drop=True)

    # 2. Coerce all non-time columns to numeric
    for col in df.columns:
        if col != time_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Handle Target Column Logic
    if target_column:
        print(f"User specified target column: '{target_column}'")
        if target_column not in df.columns:
            raise ValueError(f"The selected column '{target_column}' was not found in the file.")
        if target_column not in df.select_dtypes(include=np.number).columns:
            raise ValueError(f"The selected column '{target_column}' is not numeric and cannot be predicted.")
    else:
        # Infer target only if one was not provided
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found to use as a prediction target.")
        target_column = numeric_cols[-1]
        print(f"Inferred target column: '{target_column}'")

    # 4. Final Cleanup
    # Drop rows where the final target is NaN, which is crucial
    df.dropna(subset=[target_column], inplace=True)
    if df.empty:
        raise ValueError(f"No valid data remains for the target column '{target_column}' after cleaning.")
        
    df.drop(columns=[col for col in df.columns if df[col].nunique(dropna=False) <= 1], inplace=True, errors='ignore')
    
    return df, time_col, target_column

def create_lag_features(df, target_column, lags):
    df_lagged = df.copy()
    for lag in lags:
        df_lagged[f'{target_column}_lag_{lag}'] = df_lagged[target_column].shift(lag)
    return df_lagged

def create_cyclical_features(df, time_col):
    df_cyc = df.copy()
    if time_col and pd.api.types.is_datetime64_any_dtype(df_cyc[time_col]):
        df_cyc['month_sin'] = np.sin(2 * np.pi * df_cyc[time_col].dt.month / 12)
        df_cyc['month_cos'] = np.cos(2 * np.pi * df_cyc[time_col].dt.month / 12)
    return df_cyc

def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)


# --- 4. Helper Classes (Preprocessor and PyTorch Components) ---

class FeaturePreprocessor:
    def __init__(self, target_column, time_column=None):
        self.target_column = target_column
        self.time_column = time_column
        self.preprocessor = None

    def fit(self, df):
        features_df = df.drop(columns=[c for c in [self.target_column, self.time_column] if c in df.columns], errors='ignore')
        num_cols = features_df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = features_df.select_dtypes(exclude=np.number).columns.tolist()
        
        self.preprocessor = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
        ], remainder='passthrough').fit(features_df)
        return self

    def transform(self, df):
        y = df[self.target_column].values if self.target_column in df.columns else None
        features_df = df.drop(columns=[c for c in [self.target_column, self.time_column] if c in df.columns], errors='ignore')
        return self.preprocessor.transform(features_df), y

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_heads, embedding_dim, dropout):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, embedding_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQUENCE_LENGTH, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(embedding_dim, output_dim)
    def forward(self, src):
        if src.dim() == 2: src = src.unsqueeze(1)
        src = self.input_embedding(src) + self.pos_encoder
        transformer_output = self.transformer_encoder(src)
        return self.output_head(transformer_output[:, -1, :])

# --- 5. Main Pipeline Function (Callable from Django) ---

# --- New Lightweight Analysis Function ---
def analyze_file_columns(file_path):
    try:
        # The new loader handles inference correctly when target_column is None
        df, time_col, _ = load_and_detect_data(file_path, target_column=None)

        if not time_col:
            return {"status": "error", "message": "The uploaded file is not a valid time series. No date/time column was detected."}
        
        potential_targets = [col for col in df.select_dtypes(include=np.number).columns.tolist() if col != time_col]
        
        if not potential_targets:
            return {"status": "error", "message": "No numeric columns suitable for prediction were found in the dataset."}

        return { "status": "success", "time_column": time_col, "available_columns": potential_targets }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- New Forecasting Pipeline Function ---
# --- NEW ROBUST Forecasting Pipeline Function with Log Transformation ---
def run_forecasting_pipeline(file_path, target_column):
    print(f"--- Starting Forecasting Pipeline for target: {target_column} ---")
    try:
        # 1. Load data
        df, time_col, target_col = load_and_detect_data(file_path, target_column=target_column)
        
        forecast_steps = int(len(df) * 0.20)
        print(f"Training on {len(df)} rows, forecasting the next {forecast_steps} steps.")
        
        # --- !!! NEW: LOG TRANSFORMATION STEP !!! ---
        # We use log1p which handles zeros gracefully (log(1+x))
        print(f"Applying log transformation to target column '{target_col}' to handle volatility.")
        df[target_col] = np.log1p(df[target_col])

        # 2. Train Model on 100% of the log-transformed data
        best_params = {'lr': 0.001, 'n_layers': 2, 'n_heads': 4, 'embedding_dim': 128, 'dropout': 0.2}
        
        train_df_eng = create_cyclical_features(create_lag_features(df, target_col, LAG_FEATURES), time_col).dropna()
        
        if len(train_df_eng) <= SEQUENCE_LENGTH:
            raise ValueError(
                f"The dataset is too small for the current model configuration. "
                f"After creating features, only {len(train_df_eng)} rows remained, "
                f"but a minimum of {SEQUENCE_LENGTH + 1} are required. "
                f"Please try a larger dataset."
            )

        preprocessor = FeaturePreprocessor(target_col, time_col).fit(train_df_eng)
        X_train_proc, y_train_raw = preprocessor.transform(train_df_eng)
        X_train, y_train = create_sequences(X_train_proc, y_train_raw, SEQUENCE_LENGTH)

        model_params = {k: v for k, v in best_params.items() if k != 'lr'}
        model = TabularTransformer(input_dim=X_train.shape[-1], output_dim=1, **model_params).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
        criterion = nn.MSELoss()
        train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=32, shuffle=True)
        
        print("Training model on full log-transformed dataset...")
        model.train()
        for epoch in range(50):
            for X_b, y_b in train_loader:
                optimizer.zero_grad(); loss = criterion(model(X_b.to(DEVICE)), y_b.to(DEVICE)); loss.backward(); optimizer.step()

        # 3. Autoregressive Forecasting (in log space)
        print("Starting autoregressive forecasting...")
        model.eval()
        log_future_predictions = []
        rolling_df = df.tail(SEQUENCE_LENGTH + max(LAG_FEATURES)).copy()
        freq = pd.infer_freq(df[time_col]) or 'D'
        last_date = df[time_col].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=freq)[1:]

        with torch.no_grad():
            for i in range(forecast_steps):
                df_eng = create_cyclical_features(create_lag_features(rolling_df, target_col, LAG_FEATURES), time_col)
                sequence_to_predict_on = df_eng.tail(SEQUENCE_LENGTH)
                X_proc, _ = preprocessor.transform(sequence_to_predict_on)
                # Prediction is in log space
                log_prediction = model(torch.tensor(X_proc, dtype=torch.float32).unsqueeze(0).to(DEVICE)).squeeze().item()
                log_future_predictions.append(log_prediction)
                new_row = {col: [np.nan] for col in df.columns}
                new_row[time_col] = [future_dates[i]]
                new_row[target_col] = [log_prediction] # Add the log prediction back for the next step
                rolling_df = pd.concat([rolling_df, pd.DataFrame(new_row)], ignore_index=True)

        # --- !!! NEW: INVERSE TRANSFORMATION STEP !!! ---
        # Convert predictions AND historical data back to the original scale for the chart
        print("Applying inverse transformation to format results for frontend...")
        future_predictions = np.expm1(log_future_predictions)
        df[target_col] = np.expm1(df[target_col])
        
        # 4. Format results
        historical_df = df[[time_col, target_col]].rename(columns={target_col: 'actual'})
        forecast_df = pd.DataFrame({time_col: future_dates, 'forecast': future_predictions})
        combined_chart_data = pd.concat([historical_df, forecast_df], ignore_index=True)
        
        json_compliant_chart_data = combined_chart_data.astype(object).where(pd.notnull(combined_chart_data), None)
        json_compliant_chart_data[time_col] = pd.to_datetime(json_compliant_chart_data[time_col]).dt.strftime('%Y-%m-%d')
        forecast_df[time_col] = forecast_df[time_col].dt.strftime('%Y-%m-%d')
        
        return {
            "status": "success",
            "time_column": time_col,
            "target_column": target_column,
            "chart_data": json_compliant_chart_data.to_dict(orient='records'),
            "forecast_table_data": forecast_df.to_dict(orient='records')
        }
    except Exception as e:
        print(f"ERROR in forecasting pipeline: {e}")
        return {"status": "error", "message": str(e)}