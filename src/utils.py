# src/utils.py
import pandas as pd
import joblib
from scipy.spatial import cKDTree
import numpy as np
import streamlit as st
from .config import * # Import constants

@st.cache_resource
def load_model_resources():
    """Loads the trained model and the species encoder."""
    try:
        model = joblib.load(MODEL_FILENAME)
        encoder = joblib.load(ENCODER_FILENAME)
        return model, encoder
    except FileNotFoundError:
        return None, None

def calculate_competition_index(df, radius=COMPETITION_RADIUS):
    """
    Calculates Hegyi's Competition Index (CI) using Vectorized NumPy operations.
    CI = Sum( (Neighbor_DBH / Subject_DBH) / Distance )
    
    Performance Note:
    This vectorized version is significantly faster than iterating through rows.
    """
    # 1. Prepare Data
    # Filter valid trees (must have Coords and DBH). 
    # D19 is used as the current diameter for competition calculations.
    valid_trees = df.dropna(subset=['XCO', 'YCO', 'D19']).copy()
    valid_trees.reset_index(drop=True, inplace=True)
    
    # If no valid trees, return early
    if valid_trees.empty:
        df['Competition_Index'] = 0.0
        return df

    # Extract numpy arrays for high-speed processing
    coords = valid_trees[['XCO', 'YCO']].values
    dbh = valid_trees['D19'].values
    n_trees = len(valid_trees)
    
    # 2. Build Spatial Tree
    tree = cKDTree(coords)
    
    # 3. Find Pairs (Vectorized)
    # query_pairs finds all unique pairs (i, j) where dist(i,j) < r and i < j
    # This avoids calculating the full distance matrix (N*N), saving huge memory.
    pairs_set = tree.query_pairs(r=radius)
    
    if not pairs_set:
        valid_trees['Competition_Index'] = 0.0
    else:
        # Convert set of pairs to a NumPy array: shape (N_pairs, 2)
        pairs = np.array(list(pairs_set))
        
        # 4. Calculate Distances & Interactions
        idx_a = pairs[:, 0]
        idx_b = pairs[:, 1]
        
        # Get coordinates and calculate Euclidean distances
        vec_diff = coords[idx_a] - coords[idx_b]
        dists = np.linalg.norm(vec_diff, axis=1)
        
        # Avoid division by zero (unlikely with query_pairs > 0, but good practice)
        dists = np.maximum(dists, 1e-6)
        
        # Get DBH values
        dbh_a = dbh[idx_a]
        dbh_b = dbh[idx_b]
        
        # 5. Compute Bidirectional CI
        # If A and B are neighbors:
        # A exerts pressure on B: (DBH_A / DBH_B) / dist
        pressure_on_b = (dbh_a / dbh_b) / dists
        
        # B exerts pressure on A: (DBH_B / DBH_A) / dist
        pressure_on_a = (dbh_b / dbh_a) / dists
        
        # 6. Aggregate Scores
        # Sum the pressures for each tree index
        # np.bincount is a very fast way to sum values at specific indices
        ci_a = np.bincount(idx_a, weights=pressure_on_a, minlength=n_trees)
        ci_b = np.bincount(idx_b, weights=pressure_on_b, minlength=n_trees)
        
        # Total CI is the sum of pressure received from all neighbors
        valid_trees['Competition_Index'] = ci_a + ci_b
    
    # 7. Merge back to original DataFrame
    # If column exists, drop it to avoid duplication during merge
    if 'Competition_Index' in df.columns:
        df = df.drop(columns=['Competition_Index'])
        
    return df.merge(valid_trees[['TAG', 'Competition_Index']], on='TAG', how='left')

@st.cache_data
def load_and_process_data(csv_path):
    try:
        # 1. Load Data
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin1')
            
        # 2. Clean Names & Data
        df.columns = df.columns.str.strip()
        df.rename(columns=lambda x: x.replace('ï»¿', ''), inplace=True)
        if 'SP' in df.columns:
            df['SP'] = df['SP'].astype(str).str.strip()
        
        # Replace 0s with NA to prevent division by zero in CI calculations
        df.replace(0, pd.NA, inplace=True)

        # 3. Basic Spatial Features
        spatial_df = df.dropna(subset=['XCO', 'YCO']).copy()
        if not spatial_df.empty:
            tree = cKDTree(spatial_df[['XCO', 'YCO']].values)
            distances, _ = tree.query(spatial_df[['XCO', 'YCO']].values, k=2)
            spatial_df['Nearest_Neighbor_Dist'] = distances[:, 1]
            
            counts = tree.query_ball_point(spatial_df[['XCO', 'YCO']].values, r=DENSITY_RADIUS, return_length=True)
            spatial_df['Local_Density'] = counts - 1
            
            df = df.merge(spatial_df[['TAG', 'Nearest_Neighbor_Dist', 'Local_Density']], on='TAG', how='left')

        # 4. ADVANCED: Calculate Competition Index (Now Vectorized!)
        df = calculate_competition_index(df)

        # 5. Prepare for Prediction
        df.dropna(subset=['D17', 'D19', 'Competition_Index'], inplace=True)
        df['GROWTH1719'] = df['D19'] - df['D17']
        
        return df
    except FileNotFoundError:
        st.error(f"File not found: {csv_path}")
        return None

def run_predictions(df, model, encoder):
    """Runs predictions using Model + Encoder."""
    # Encode Species
    # We use 'map' with the classes_ from the encoder to be safe
    species_map = {species: i for i, species in enumerate(encoder.classes_)}
    df['SP_Encoded'] = df['SP'].map(species_map).fillna(-1) # -1 for unknown species
    
    features = ['D19', 'GROWTH1719', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded']
    
    # Drop rows with missing features
    df_pred = df[features].dropna()
    
    if not df_pred.empty:
        df_pred['Predicted_D21'] = model.predict(df_pred)
        df_pred['Predicted_Growth'] = df_pred['Predicted_D21'] - df_pred['D19']
        return df.join(df_pred[['Predicted_D21', 'Predicted_Growth']])
    
    return df