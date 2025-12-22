# train_model.py (Advanced Version)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
from src.utils import calculate_competition_index # Re-use logic!
from src.config import *
import warnings

warnings.filterwarnings("ignore") # Clean output

print("Starting Advanced Model Training...")

# 1. Load & Clean
try:
    df = pd.read_csv(DATA_FILENAME, encoding='utf-8-sig')
except:
    df = pd.read_csv(DATA_FILENAME, encoding='latin1')

df.columns = df.columns.str.strip()
df.rename(columns=lambda x: x.replace('ï»¿', ''), inplace=True)
if 'SP' in df.columns:
    df['SP'] = df['SP'].astype(str).str.strip()
df.replace(0, pd.NA, inplace=True)

print(f"   Loaded {len(df)} records.")

# 2. Spatial Features (Density, Distance, CI)
print("   Calculating Spatial Metrics & Competition Index...")
# We use the function we just wrote in src/utils to ensure consistency
try:
    # Basic Spatial
    from scipy.spatial import cKDTree
    spatial_df = df.dropna(subset=['XCO', 'YCO']).copy()
    coords = spatial_df[['XCO', 'YCO']].values
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    spatial_df['Nearest_Neighbor_Dist'] = dists[:, 1]
    counts = tree.query_ball_point(coords, r=DENSITY_RADIUS, return_length=True)
    spatial_df['Local_Density'] = counts - 1
    df = df.merge(spatial_df[['TAG', 'Nearest_Neighbor_Dist', 'Local_Density']], on='TAG', how='left')
    
    # Advanced CI
    df = calculate_competition_index(df, radius=COMPETITION_RADIUS)
except Exception as e:
    print(f"❌ Error calculating spatial features: {e}")
    sys.exit()

# 3. Feature Engineering
df.dropna(subset=['D17', 'D19', 'Competition_Index'], inplace=True)
df['GROWTH1719'] = df['D19'] - df['D17']

# 4. Species Encoding
print("   Encoding Species...")
encoder = LabelEncoder()
df['SP_Encoded'] = encoder.fit_transform(df['SP'])

features = ['D19', 'GROWTH1719', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded']
target = 'D21'

df_train = df.dropna(subset=[target] + features)
X = df_train[features]
y = df_train[target]

# 5. Train & Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- 📊 Model Performance ---")
print(f"   RMSE: {rmse:.4f} cm")
print(f"   R²:   {r2:.4f}")

# 6. Save Everything
print("\n   💾 Saving Model and Encoder...")
model.fit(X, y) # Retrain on full data
joblib.dump(model, MODEL_FILENAME)
joblib.dump(encoder, ENCODER_FILENAME)

print("✅ Done! Ready to launch app.")