# train_model.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier  # <--- XGBoost Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.spatial import cKDTree
from src.config import *
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

print(f"🌲 Training System (XGBoost) on: {DATA_FILENAME}")

# ==========================================
# 1. LOAD & CLEAN
# ==========================================
try:
    df = pd.read_csv(DATA_FILENAME, encoding='utf-8-sig')
except:
    df = pd.read_csv(DATA_FILENAME, encoding='latin1')

df.columns = df.columns.str.strip()

# Clean Species Column
if COL_SPECIES in df.columns:
    df[COL_SPECIES] = df[COL_SPECIES].astype(str).str.strip()

# Clean Numeric Columns (History, Current, Target)
for c in [COL_HISTORY, COL_CURRENT, COL_TARGET]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("   Processing Spatial Features...")

# We need trees that were ALIVE at the start (Current > 0)
df_valid = df[df[COL_CURRENT] > 0].copy()

# Drop rows with invalid coordinates
df_valid = df_valid.dropna(subset=[COL_X, COL_Y])

# A. Past Growth (History -> Current)
# If history is 0, we assume 0 growth for calculation
df_valid['GROWTH_HIST'] = df_valid[COL_CURRENT] - df_valid[COL_HISTORY]
df_valid.loc[df_valid[COL_HISTORY] == 0, 'GROWTH_HIST'] = 0 

# B. Spatial Metrics (Density & Distance)
coords = df_valid[[COL_X, COL_Y]].values
tree = cKDTree(coords)

dists, _ = tree.query(coords, k=2) # k=2 because k=1 is the tree itself
df_valid['Nearest_Neighbor_Dist'] = dists[:, 1]

counts = tree.query_ball_point(coords, r=DENSITY_RADIUS, return_length=True)
df_valid['Local_Density'] = counts - 1

# C. Competition Index
idx_map = []
neighbor_indices = tree.query_ball_point(coords, r=COMPETITION_RADIUS)
dbh_values = df_valid[COL_CURRENT].values

for i, neighbors in enumerate(neighbor_indices):
    subject_dbh = dbh_values[i]
    ci = 0
    for n_idx in neighbors:
        if i == n_idx: continue
        dist = np.linalg.norm(coords[i] - coords[n_idx])
        if dist > 0:
            # Neighbor size matters, even if it dies later
            ci += (dbh_values[n_idx] / subject_dbh) / dist
    idx_map.append(ci)
df_valid['Competition_Index'] = idx_map

# D. Encode Species
encoder = LabelEncoder()
df_valid['SP_Encoded'] = encoder.fit_transform(df_valid[COL_SPECIES])

# Define Feature Vector
features = [COL_CURRENT, 'GROWTH_HIST', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded']

# ==========================================
# 3. TRAIN MORTALITY MODEL (XGBoost Classifier)
# ==========================================
print("\n💀 Training Mortality Risk Model (XGBoost)...")
# Target: 1 if Died (Target=0), 0 if Survived (Target>0)
df_valid['IS_DEAD'] = (df_valid[COL_TARGET] == 0).astype(int)

X = df_valid[features]
y_mort = df_valid['IS_DEAD']

# Handle Class Imbalance (Dead trees are usually minority)
num_dead = y_mort.sum()
num_alive = len(y_mort) - num_dead
scale_pos_weight = num_alive / num_dead if num_dead > 0 else 1.0

X_train, X_test, y_train, y_test = train_test_split(X, y_mort, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# XGBClassifier
clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=scale_pos_weight, # Handles imbalance automatically
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='logloss'
)
clf.fit(X_train, y_train)

y_pred_class = clf.predict(X_test)
print(f"   Accuracy: {accuracy_score(y_test, y_pred_class):.4f}")

# ==========================================
# 4. TRAIN GROWTH MODEL (XGBoost Regressor)
# ==========================================
print("\n📈 Training Growth Model (XGBoost)...")
# Train ONLY on trees that survived
df_survived = df_valid[df_valid['IS_DEAD'] == 0].copy()

X_grow = df_survived[features]
y_grow = df_survived[COL_TARGET]

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_grow, y_grow, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# XGBRegressor
reg = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
reg.fit(X_train_g, y_train_g)

y_pred_g = reg.predict(X_test_g)
rmse = np.sqrt(mean_squared_error(y_test_g, y_pred_g))
print(f"   RMSE: {rmse:.4f} cm")

# ==========================================
# 5. SAVE EVERYTHING
# ==========================================
print("\n💾 Saving Artifacts...")
# Retrain on full data for best performance
clf.fit(X, y_mort)
reg.fit(X_grow, y_grow)

# Save using joblib (XGBoost models are picklable)
joblib.dump(reg, MODEL_FILENAME)
joblib.dump(clf, MORTALITY_MODEL_FILENAME)
joblib.dump(encoder, ENCODER_FILENAME)

print("✅ XGBoost Upgrade Complete. Models saved successfully.")