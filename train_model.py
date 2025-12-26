# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.spatial import cKDTree
from src.config import *
import warnings

warnings.filterwarnings("ignore")

print(f"🌲 Training Model on: {DATA_FILENAME}")
print(f"   Mapping: {COL_HISTORY} -> {COL_CURRENT} (Input) to Predict {COL_TARGET}")

# 1. Load Data
try:
    df = pd.read_csv(DATA_FILENAME, encoding='utf-8-sig')
except:
    df = pd.read_csv(DATA_FILENAME, encoding='latin1')

df.columns = df.columns.str.strip()
if COL_SPECIES in df.columns:
    df[COL_SPECIES] = df[COL_SPECIES].astype(str).str.strip()

# Cleanup Numerics
for c in [COL_HISTORY, COL_CURRENT, COL_TARGET]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c].replace(0, np.nan, inplace=True)

# 2. Prepare Training Features
# We need rows that have History, Current, AND Target (to learn from)
df = df.dropna(subset=[COL_X, COL_Y, COL_HISTORY, COL_CURRENT, COL_TARGET]).copy()

# A. Past Growth
df['GROWTH_HIST'] = df[COL_CURRENT] - df[COL_HISTORY]

# B. Spatial Metrics (Based on Current D05)
coords = df[[COL_X, COL_Y]].values
tree = cKDTree(coords)

dists, _ = tree.query(coords, k=2)
df['Nearest_Neighbor_Dist'] = dists[:, 1]

counts = tree.query_ball_point(coords, r=DENSITY_RADIUS, return_length=True)
df['Local_Density'] = counts - 1

# C. Competition Index
print("   Calculating Competition Index...")
idx_map = []
neighbor_indices = tree.query_ball_point(coords, r=COMPETITION_RADIUS)
dbh_values = df[COL_CURRENT].values

for i, neighbors in enumerate(neighbor_indices):
    subject_dbh = dbh_values[i]
    ci = 0
    for n_idx in neighbors:
        if i == n_idx: continue
        dist = np.linalg.norm(coords[i] - coords[n_idx])
        if dist > 0:
            ci += (dbh_values[n_idx] / subject_dbh) / dist
    idx_map.append(ci)
df['Competition_Index'] = idx_map

# 3. Train
print("   Fitting Model...")
encoder = LabelEncoder()
df['SP_Encoded'] = encoder.fit_transform(df[COL_SPECIES])

# Exact same features as run_predictions in utils.py
features = [COL_CURRENT, 'GROWTH_HIST', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded']

X = df[features]
y = df[COL_TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} cm")

# Save
model.fit(X, y)
joblib.dump(model, MODEL_FILENAME)
joblib.dump(encoder, ENCODER_FILENAME)
print("✅ Model Saved.")