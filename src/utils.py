# src/utils.py
import pandas as pd
import joblib
from scipy.spatial import cKDTree
import numpy as np
import streamlit as st
from .config import *

@st.cache_resource
def load_model_resources():
    try:
        model_grow = joblib.load(MODEL_FILENAME)
        try:
            model_mort = joblib.load(MORTALITY_MODEL_FILENAME)
        except FileNotFoundError:
            model_mort = None
        encoder = joblib.load(ENCODER_FILENAME)
        return model_grow, model_mort, encoder
    except FileNotFoundError:
        return None, None, None

def calculate_competition_index(df, radius_meters=COMPETITION_RADIUS):
    """
    Calculates Hegyi's Competition Index with Degree-to-Meter conversion.
    """
    if COL_CURRENT not in df.columns: return df

    # Filter valid trees
    valid_trees = df.dropna(subset=[COL_X, COL_Y, COL_CURRENT]).copy().reset_index(drop=True)
    
    # 1. Coordinate Setup (Degrees)
    coords = valid_trees[[COL_X, COL_Y]].values
    dbh = valid_trees[COL_CURRENT].values
    
    # 2. Convert Radius: Meters -> Degrees for the search
    # 1 degree approx 111,111 meters. So 6m becomes ~0.000054 degrees
    DEG_PER_METER = 1 / 111111.0
    radius_deg = radius_meters * DEG_PER_METER
    
    # 3. Build Tree & Query
    tree = cKDTree(coords)
    neighbors_list = tree.query_ball_point(coords, r=radius_deg)
    
    ci_scores = []
    for i, neighbors in enumerate(neighbors_list):
        subject_dbh = dbh[i]
        subject_ci = 0
        for n_idx in neighbors:
            if i == n_idx: continue
            
            # Calculate raw distance in degrees
            dist_deg = np.linalg.norm(coords[i] - coords[n_idx])
            
            # 4. CRITICAL FIX: Convert distance to Meters for the formula
            dist_m = dist_deg * 111111.0
            
            if dist_m > 0.1: # Avoid division by zero
                n_dbh = dbh[n_idx]
                if n_dbh > 0 and subject_dbh > 0:
                    # Hegyi Formula: (D_j / D_i) / Dist_ij
                    subject_ci += (n_dbh / subject_dbh) / dist_m
                    
        ci_scores.append(subject_ci)
        
    valid_trees['Competition_Index'] = ci_scores
    return df.merge(valid_trees[[COL_ID, 'Competition_Index']], on=COL_ID, how='left')

@st.cache_data(show_spinner=True)
def load_and_process_data(csv_path, min_dbh_limit=DEFAULT_MIN_DBH):
    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(csv_path, encoding='latin1')

        df.columns = df.columns.str.strip()
        
        # Text Cleaning
        if COL_SPECIES in df.columns: 
            df[COL_SPECIES] = df[COL_SPECIES].astype(str).str.strip()
        if COL_SPECIES_GRP in df.columns: 
            df[COL_SPECIES_GRP] = df[COL_SPECIES_GRP].astype(str).str.strip()
        
        # Numeric Cleaning
        # Important: Your coordinates are likely already valid, but check for 0s
        for c in [COL_HISTORY, COL_CURRENT, COL_TARGET]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Filter
        if COL_CURRENT in df.columns:
            df = df[df[COL_CURRENT] >= min_dbh_limit]

        # Features
        if COL_HISTORY in df.columns and COL_CURRENT in df.columns:
            hist_vals = df[COL_HISTORY].fillna(df[COL_CURRENT])
            df['GROWTH_HIST'] = df[COL_CURRENT] - hist_vals

        # Spatial Features
        # Note: We calculate these using degrees, which is fine for relative ranking,
        # but technically Density Radius should also be converted.
        spatial_df = df.dropna(subset=[COL_X, COL_Y, COL_CURRENT]).copy()
        
        if not spatial_df.empty:
            coords = spatial_df[[COL_X, COL_Y]].values
            tree = cKDTree(coords)
            
            # Use same conversion for consistency if needed, but for now simple query is okay
            # Note: 5 meters in degrees is approx 0.000045
            radius_deg = DENSITY_RADIUS * (1/111111.0)
            
            dists, _ = tree.query(coords, k=2)
            spatial_df['Nearest_Neighbor_Dist'] = dists[:, 1] * 111111.0 # Convert to meters
            
            counts = tree.query_ball_point(coords, r=radius_deg, return_length=True)
            spatial_df['Local_Density'] = counts - 1
            
            df = df.merge(spatial_df[[COL_ID, 'Nearest_Neighbor_Dist', 'Local_Density']], on=COL_ID, how='left')

        # Competition Index (Now fixed)
        df = calculate_competition_index(df)

        required_cols = [COL_CURRENT, 'GROWTH_HIST', 'Competition_Index']
        df.dropna(subset=required_cols, inplace=True)
        
        return df

    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None

def run_predictions(df, model_grow, model_mort, encoder):
    if 'SP_Encoded' not in df.columns:
        known_classes = encoder.classes_
        map_dict = {sp: i for i, sp in enumerate(known_classes)}
        df['SP_Encoded'] = df[COL_SPECIES].map(map_dict).fillna(-1) 
    
    features = [COL_CURRENT, 'GROWTH_HIST', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded']
    
    pred_df = df[features].dropna()
    pred_df = pred_df[pred_df['SP_Encoded'] != -1]
    
    if not pred_df.empty:
        predicted_future = model_grow.predict(pred_df)
        corrected_future = np.maximum(predicted_future, pred_df[COL_CURRENT])
        
        df.loc[pred_df.index, 'Predicted_Size'] = corrected_future 
        df.loc[pred_df.index, 'Predicted_Growth'] = corrected_future - pred_df[COL_CURRENT]
        
        if model_mort is not None:
            risk_probs = model_mort.predict_proba(pred_df)[:, 1] 
            df.loc[pred_df.index, 'Mortality_Risk'] = risk_probs
        else:
            df['Mortality_Risk'] = 0.0
        
    return df