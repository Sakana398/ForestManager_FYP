# train_model.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.spatial import cKDTree
from src.config import *
import warnings

warnings.filterwarnings("ignore")

print(f"🌲 Training System (Growth Increment Mode) on: {DATA_FILENAME}")

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
print("   Processing Spatial Features...")
df_valid = df[df[COL_CURRENT] > 0].copy()
df_valid = df_valid.dropna(subset=[COL_X, COL_Y])

# Features
df_valid['GROWTH_HIST'] = df_valid[COL_CURRENT] - df_valid[COL_HISTORY]
df_valid.loc[df_valid[COL_HISTORY] == 0, 'GROWTH_HIST'] = 0 

# Spatial
coords = df_valid[[COL_X, COL_Y]].values
tree = cKDTree(coords)

dists, _ = tree.query(coords, k=2) 
df_valid['Nearest_Neighbor_Dist'] = dists[:, 1] * 111111 # Convert deg to meters

# Density
radius_deg = DENSITY_RADIUS * (1/111111)
counts = tree.query_ball_point(coords, r=radius_deg, return_length=True)
df_valid['Local_Density'] = counts - 1

# Competition Index (Fixed Meters Logic)
idx_map = []
radius_ci_deg = COMPETITION_RADIUS * (1/111111)
neighbor_indices = tree.query_ball_point(coords, r=radius_ci_deg)
dbh_values = df_valid[COL_CURRENT].values

for i, neighbors in enumerate(neighbor_indices):
    subject_dbh = dbh_values[i]
    ci = 0
    for n_idx in neighbors:
        if i == n_idx: continue
        d_deg = np.linalg.norm(coords[i] - coords[n_idx])
        d_m = d_deg * 111111
        if d_m > 0.1:
            ci += (dbh_values[n_idx] / subject_dbh) / d_m
    idx_map.append(ci)
df_valid['Competition_Index'] = idx_map

# Encoder
encoder = LabelEncoder()
df_valid['SP_Encoded'] = encoder.fit_transform(df_valid[COL_SPECIES])

features = [COL_CURRENT, 'GROWTH_HIST', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded']

# ==========================================
# 3. TRAIN MORTALITY
# ==========================================
print("\n💀 Training Mortality Risk Model...")
df_valid['IS_DEAD'] = (df_valid[COL_TARGET] == 0).astype(int)
X = df_valid[features]
y_mort = df_valid['IS_DEAD']

num_dead = y_mort.sum()
weight = (len(y_mort) - num_dead) / num_dead if num_dead > 0 else 1.0

clf = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    scale_pos_weight=weight, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'
)
clf.fit(X, y_mort) # Fit on full data for app

# ==========================================
# 4. TRAIN GROWTH (INCREMENT MODE)
# ==========================================
print("\n📈 Training Growth Model (Predicting Increment)...")
df_survived = df_valid[df_valid['IS_DEAD'] == 0].copy()

X_grow = df_survived[features]

# --- KEY CHANGE: TARGET IS GROWTH, NOT TOTAL SIZE ---
y_grow_total = df_survived[COL_TARGET]
y_grow_increment = y_grow_total - df_survived[COL_CURRENT] # Predict the CHANGE

# Clip negative growth (trees shouldn't shrink) to 0 for training stability
y_grow_increment = y_grow_increment.clip(lower=-0.5) 

reg = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    random_state=RANDOM_STATE, n_jobs=-1
)
reg.fit(X_grow, y_grow_increment)

# Check Importance
print("   Feature Importance:")
for name, score in zip(features, reg.feature_importances_):
    print(f"   - {name}: {score:.4f}")

# ==========================================
# 5. SAVE
# ==========================================
joblib.dump(reg, MODEL_FILENAME)
joblib.dump(clf, MORTALITY_MODEL_FILENAME)
joblib.dump(encoder, ENCODER_FILENAME)
print("\n✅ Models Updated (Growth Increment Mode).")