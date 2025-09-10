# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

RANDOM_STATE = 42

# 1) Load dataset
df = pd.read_csv("odisha_crop_data.csv", parse_dates=["sowing_date", "harvest_date"])

# 2) Feature engineering
df["harvest_days"] = (df["harvest_date"] - df["sowing_date"]).dt.days
df["sowing_doy"] = df["sowing_date"].dt.dayofyear

# Input features (only these will be provided by user)
input_features = ["district", "crop", "season", "sowing_doy"]

# Target features (all these will be predicted by the model)
target_cols = [
    "season_total_rainfall_mm", "season_avg_temp_c", "season_avg_humidity",
    "soil_pH", "soil_N_kg_ha", "soil_P_kg_ha", "soil_K_kg_ha", 
    "organic_carbon_pct", "soil_moisture_pct",
    "fertilizer_N_kg_ha", "fertilizer_P_kg_ha", "fertilizer_K_kg_ha",
    "yield_kg_per_ha", "harvest_days"
]

# Debug: Check what columns actually exist in the dataset
print("Columns in dataset:", list(df.columns))
print("Target columns:", target_cols)

# Filter only existing columns
existing_cols = [col for col in (input_features + target_cols) if col in df.columns]
df = df[existing_cols].copy()

# Update target_cols to only include existing columns
target_cols = [col for col in target_cols if col in df.columns]
print("Target columns after filtering:", target_cols)

# Drop rows with missing target values
df = df.dropna(subset=target_cols)
print(f"Dataset shape after preprocessing: {df.shape}")

X = df[input_features]
y = df[target_cols]

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X columns: {list(X.columns)}")
print(f"y columns: {list(y.columns)}")

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 4) Preprocessing
# Only categorical features as inputs
categorical_features = ["district", "crop", "season"]
numeric_features = ["sowing_doy"]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_features),
    ("num", numeric_transformer, numeric_features)
], remainder="drop")

# 5) Model: Multi-output RandomForest
base_reg = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
multi_reg = MultiOutputRegressor(base_reg)

pipeline = Pipeline(steps=[
    ("preproc", preprocessor),
    ("model", multi_reg)
])

# 6) Train model
print("Training model...")
pipeline.fit(X_train, y_train)
print("Model training completed!")

# 7) Evaluation
y_pred = pipeline.predict(X_test)

def print_metrics(y_true, y_pred, name):
    for i, col in enumerate(y_true.columns):
        mse = mean_squared_error(y_true[col], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true[col], y_pred[:, i])
        r2 = r2_score(y_true[col], y_pred[:, i])
        print(f"{name} - {col}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

print_metrics(y_test, y_pred, "Test Set")

# 8) Save pipeline
joblib.dump(pipeline, "odisha_crop_pipeline.joblib")
print("✅ Model trained and saved as odisha_crop_pipeline.joblib")

# Print model info for API
print(f"\nModel info for API:")
print(f"Number of outputs: {len(target_cols)}")
print(f"Output columns: {target_cols}")