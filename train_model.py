# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.spatial import cKDTree
from src.config import *
import warnings

warnings.filterwarnings("ignore")

print(f"🌲 Training System on: {DATA_FILENAME}")

# 1. LOAD & CLEAN
try:
    df = pd.read_csv(DATA_FILENAME, encoding='utf-8-sig')
except:
    df = pd.read_csv(DATA_FILENAME, encoding='latin1')

df.columns = df.columns.str.strip()
if COL_SPECIES in df.columns:
    df[COL_SPECIES] = df[COL_SPECIES].astype(str).str.strip()

for c in [COL_HISTORY, COL_CURRENT, COL_TARGET]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0) # Fill NaN with 0 for logic checks

# 2. FEATURE ENGINEERING (Common for both models)
# We need trees that were ALIVE at the start (Current > 0)
df_valid = df[df[COL_CURRENT] > 0].copy()

# Filter checks
df_valid = df_valid.dropna(subset=[COL_X, COL_Y])

# A. Past Growth (History -> Current)
# If history is 0, we assume 0 growth or handle gracefully
df_valid['GROWTH_HIST'] = df_valid[COL_CURRENT] - df_valid[COL_HISTORY]
# Fix weird cases where history was missing (0) resulting in massive growth
df_valid.loc[df_valid[COL_HISTORY] == 0, 'GROWTH_HIST'] = 0 

# B. Spatial Metrics
coords = df_valid[[COL_X, COL_Y]].values
tree = cKDTree(coords)

dists, _ = tree.query(coords, k=2)
df_valid['Nearest_Neighbor_Dist'] = dists[:, 1]
counts = tree.query_ball_point(coords, r=DENSITY_RADIUS, return_length=True)
df_valid['Local_Density'] = counts - 1

# C. Competition Index
print("   Calculating Competition Index...")
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

# FEATURES LIST
features = [COL_CURRENT, 'GROWTH_HIST', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded']

# ==========================================
# 3. TRAIN MORTALITY MODEL (Classifier)
# ==========================================
print("\n💀 Training Mortality Risk Model...")
# Target: 1 if Died (Target=0), 0 if Survived (Target>0)
df_valid['IS_DEAD'] = (df_valid[COL_TARGET] == 0).astype(int)

X = df_valid[features]
y_mort = df_valid['IS_DEAD']

X_train, X_test, y_train, y_test = train_test_split(X, y_mort, test_size=TEST_SIZE, random_state=RANDOM_STATE)

clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred_class = clf.predict(X_test)
print(f"   Accuracy: {accuracy_score(y_test, y_pred_class):.4f}")

# ==========================================
# 4. TRAIN GROWTH MODEL (Regressor)
# ==========================================
print("\n📈 Training Growth Model...")
# We only train growth on trees that SURVIVED
df_survived = df_valid[df_valid['IS_DEAD'] == 0].copy()

X_grow = df_survived[features]
y_grow = df_survived[COL_TARGET]

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_grow, y_grow, test_size=TEST_SIZE, random_state=RANDOM_STATE)

reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
reg.fit(X_train_g, y_train_g)

y_pred_g = reg.predict(X_test_g)
rmse = np.sqrt(mean_squared_error(y_test_g, y_pred_g))
print(f"   RMSE: {rmse:.4f} cm")

# ==========================================
# 5. SAVE EVERYTHING
# ==========================================
# Retrain on full data
clf.fit(X, y_mort)
reg.fit(X_grow, y_grow)

joblib.dump(reg, MODEL_FILENAME)
joblib.dump(clf, MORTALITY_MODEL_FILENAME)
joblib.dump(encoder, ENCODER_FILENAME)
print("\n✅ All Models Saved Successfully.")