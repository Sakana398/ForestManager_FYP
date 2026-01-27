# train_model.py
import pandas as pd
import numpy as np
import joblib
import warnings
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from src.config import *
from src.utils import add_spatial_features, standardize_coordinates

warnings.filterwarnings("ignore")

print(f"🌲 Training System on: {DATA_FILENAME}")

# ==========================================
# 1. LOAD & CLEAN
# ==========================================
try:
    df = pd.read_csv(DATA_FILENAME, encoding='utf-8-sig')
except:
    df = pd.read_csv(DATA_FILENAME, encoding='latin1')

df.columns = df.columns.str.strip()
if COL_SPECIES in df.columns:
    df[COL_SPECIES] = df[COL_SPECIES].astype(str).str.strip()

for c in [COL_HISTORY, COL_CURRENT, COL_TARGET]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("   Generating Features...")

# A. Basic History
df['GROWTH_HIST'] = df[COL_CURRENT] - df[COL_HISTORY]
df.loc[df[COL_HISTORY] == 0, 'GROWTH_HIST'] = 0 

# B. Spatial Features
df = standardize_coordinates(df)
df = add_spatial_features(df)

# C. Encode Species
encoder = LabelEncoder()
df['SP_Encoded'] = encoder.fit_transform(df[COL_SPECIES].astype(str))

# Feature List
features = [
    COL_CURRENT, 'GROWTH_HIST', 
    'Nearest_Neighbor_Dist', 'Local_Density', 
    'Competition_Index', 'Interaction_Vigor_Comp', 
    'SP_Encoded'
]

# ==========================================
# 3. PREPARE DATASETS (CRITICAL FIX)
# ==========================================
# Filter 1: Base Valid Data (Must have coordinates and current size)
df_base = df.dropna(subset=[COL_X, COL_Y, COL_CURRENT, 'Competition_Index']).copy()
df_base = df_base[df_base[COL_CURRENT] > 0] # Must be alive at start

print(f"   Base Dataset: {len(df_base)} trees")

# --- DATASET A: MORTALITY (Includes Dead Trees) ---
# We keep trees even if Target=0 (Dead)
df_mortality = df_base.copy()
df_mortality['IS_DEAD'] = (df_mortality[COL_TARGET] == 0).astype(int)

# --- DATASET B: GROWTH (Survivors Only) ---
# We only keep trees that survived (Target > 0) AND have realistic growth
df_growth = df_base[df_base[COL_TARGET] > 0].copy()
df_growth['Target_Growth'] = df_growth[COL_TARGET] - df_growth[COL_CURRENT]

# Remove anomalies (e.g. tree shrinking 5cm or growing 10cm in 5 years)
mask_realistic = (df_growth['Target_Growth'] > -0.5) & (df_growth['Target_Growth'] < 5.0)
df_growth = df_growth[mask_realistic]

print(f"   Training Mortality on: {len(df_mortality)} trees (Dead: {df_mortality['IS_DEAD'].sum()})")
print(f"   Training Growth on:    {len(df_growth)} trees")

# ==========================================
# 4. TRAIN MORTALITY MODEL
# ==========================================
print("\n💀 Training Mortality Risk Model...")

X_mort = df_mortality[features]
y_mort = df_mortality['IS_DEAD']

# Check if we actually have dead trees
if y_mort.sum() < 2:
    print("   ⚠️ WARNING: Not enough dead trees to train Mortality Model. Using dummy model.")
    clf = XGBClassifier(n_estimators=1) # Dummy
    clf.fit(X_mort, y_mort)
else:
    # Handle Imbalance
    num_dead = y_mort.sum()
    weight = (len(y_mort) - num_dead) / num_dead
    
    clf = XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        scale_pos_weight=weight, # Balance classes
        eval_metric='logloss',
        base_score=0.5 # Fix for some XGBoost versions
    )
    clf.fit(X_mort, y_mort)

# Evaluate
preds = clf.predict(X_mort)
acc = accuracy_score(y_mort, preds)
print(f"   - Accuracy: {acc:.2%}")

# ==========================================
# 5. TUNE GROWTH MODEL (Survivors)
# ==========================================
print("\n📈 Tuning Growth Model (Grid Search)...")

X_grow = df_growth[features]
y_grow = df_growth['Target_Growth']

if len(df_growth) > 50:
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6],
        'subsample': [0.8, 1.0]
    }

    xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1
    )

    grid.fit(X_grow, y_grow)
    best_model = grid.best_estimator_
    print(f"\n🏆 Best Parameters: {grid.best_params_}")

    # Evaluate
    preds = best_model.predict(X_grow)
    rmse = np.sqrt(mean_squared_error(y_grow, preds))
    r2 = r2_score(y_grow, preds)

    print(f"   - RMSE: {rmse:.4f} cm")
    print(f"   - R² Score: {r2:.4f}")
else:
    print("   ⚠️ Not enough data for GridSearch. Training simple model.")
    best_model = XGBRegressor(n_estimators=100)
    best_model.fit(X_grow, y_grow)

# ==========================================
# 6. SAVE ARTIFACTS
# ==========================================
joblib.dump(best_model, MODEL_FILENAME)
joblib.dump(clf, MORTALITY_MODEL_FILENAME)
joblib.dump(encoder, ENCODER_FILENAME)
print("\n✅ Optimized Models Saved.")