# compare_models.py
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from src.config import *
# [NEW] Import the centralized logic
from src.utils import add_spatial_features, standardize_coordinates

warnings.filterwarnings("ignore")

print("📊 Starting Model Evaluation Tournament...")
print(f"   Dataset: {DATA_FILENAME}")

# ==========================================
# 1. PREPARE DATA (Using Centralized Logic)
# ==========================================
try:
    df = pd.read_csv(DATA_FILENAME, encoding='utf-8-sig')
except:
    df = pd.read_csv(DATA_FILENAME, encoding='latin1')

# Clean Columns
df.columns = df.columns.str.strip()
if COL_SPECIES in df.columns:
    df[COL_SPECIES] = df[COL_SPECIES].astype(str).str.strip()

for c in [COL_HISTORY, COL_CURRENT, COL_TARGET]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# [KEY UPDATE] Use the new utility functions
print("   Generating Features via src.utils...")
df = standardize_coordinates(df)
df = add_spatial_features(df)

# ==========================================
# 2. FILTERING (Same as train_model.py)
# ==========================================
# We only want to predict growth for trees that SURVIVED and have REALISTIC data.
df['Target_Growth'] = df[COL_TARGET] - df[COL_CURRENT]

mask_valid = (
    (df[COL_CURRENT] > 0) & 
    (df[COL_TARGET] > 0) & 
    (pd.notna(df['Competition_Index']))
)

# Remove anomalies (e.g., negative growth or super-growth > 5cm)
mask_realistic = (df['Target_Growth'] > -0.5) & (df['Target_Growth'] < 5.0)

df_valid = df[mask_valid & mask_realistic].copy()

# Encode Species
encoder = LabelEncoder()
df_valid['SP_Encoded'] = encoder.fit_transform(df_valid[COL_SPECIES])

# Basic History Feature
df_valid['GROWTH_HIST'] = df_valid[COL_CURRENT] - df_valid[COL_HISTORY]
df_valid.loc[df_valid[COL_HISTORY] == 0, 'GROWTH_HIST'] = 0 

# Define Feature Sets
# [NEW] Added 'Interaction_Vigor_Comp' to the Spatial Set
feats_spatial = [
    COL_CURRENT, 'GROWTH_HIST', 'SP_Encoded', 
    'Nearest_Neighbor_Dist', 'Local_Density', 
    'Competition_Index', 'Interaction_Vigor_Comp'
]

feats_basic = [
    COL_CURRENT, 'GROWTH_HIST', 'SP_Encoded'
]

# Target
y = df_valid['Target_Growth']

print(f"   Evaluation Set: {len(df_valid)} trees.\n")

# ==========================================
# 3. DEFINE MODELS
# ==========================================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest    ": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost          ": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
}

# ==========================================
# 4. RUN TOURNAMENT
# ==========================================
print(f"{'Model':<20} | {'Features':<12} | {'RMSE':<8} | {'MAE':<8} | {'R²':<8}")
print("-" * 75)

for name, model in models.items():
    # --- Round 1: WITH Spatial Data ---
    X = df_valid[feats_spatial]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"{name:<20} | {'Spatial':<12} | {rmse:.4f}   | {mae:.4f}   | {r2:.4f}")
    
    # --- Round 2: WITHOUT Spatial Data ---
    X = df_valid[feats_basic]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"{name:<20} | {'Basic':<12} | {rmse:.4f}   | {mae:.4f}   | {r2:.4f}")
    print("-" * 75)

print("\n✅ Evaluation Complete.")