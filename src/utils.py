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
    Calculates Hegyi's Competition Index (CI).
    CI = Sum( (Neighbor_DBH / Subject_DBH) / Distance )
    """
    # We need coordinates and DBH (using D19 as current size)
    # Filter out trees with missing DBH or Coords
    valid_trees = df.dropna(subset=['XCO', 'YCO', 'D19']).copy()
    
    # Reset index to ensure alignment
    valid_trees = valid_trees.reset_index(drop=True)
    
    coords = valid_trees[['XCO', 'YCO']].values
    dbh = valid_trees['D19'].values
    
    tree = cKDTree(coords)
    
    # Find all neighbors within radius
    # returns a list of lists: [ [neighbor_idx_1, neighbor_idx_2], ... ]
    neighbors_list = tree.query_ball_point(coords, r=radius)
    
    ci_scores = []
    
    for i, neighbors in enumerate(neighbors_list):
        subject_dbh = dbh[i]
        subject_ci = 0
        
        for n_idx in neighbors:
            if i == n_idx: continue # Skip self
            
            neighbor_dbh = dbh[n_idx]
            dist = np.linalg.norm(coords[i] - coords[n_idx])
            
            if dist > 0:
                # Hegyi's Index Formula
                pressure = (neighbor_dbh / subject_dbh) / dist
                subject_ci += pressure
                
        ci_scores.append(subject_ci)
        
    valid_trees['Competition_Index'] = ci_scores
    
    # Merge back to original dataframe
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

        # 4. ADVANCED: Calculate Competition Index
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