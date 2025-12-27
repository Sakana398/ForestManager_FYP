# src/utils.py
import pandas as pd
import joblib
from scipy.spatial import cKDTree
import numpy as np
import streamlit as st
from .config import *

@st.cache_resource
def load_model_resources():
    """
    Loads the Growth Regressor, Mortality Classifier, and Species Encoder.
    Returns None if files are missing.
    """
    try:
        model_grow = joblib.load(MODEL_FILENAME)
        model_mort = joblib.load(MORTALITY_MODEL_FILENAME)
        encoder = joblib.load(ENCODER_FILENAME)
        return model_grow, model_mort, encoder
    except FileNotFoundError:
        return None, None, None

def calculate_competition_index(df, radius=COMPETITION_RADIUS):
    """
    Calculates Hegyi's Competition Index for each tree.
    CI = Sum( (Neighbor_DBH / Subject_DBH) / Distance )
    """
    if COL_CURRENT not in df.columns: return df

    # Create a subset of valid trees for spatial querying
    valid_trees = df.dropna(subset=[COL_X, COL_Y, COL_CURRENT]).copy().reset_index(drop=True)
    coords = valid_trees[[COL_X, COL_Y]].values
    dbh = valid_trees[COL_CURRENT].values
    
    # Build KDTree for fast neighbor lookup
    tree = cKDTree(coords)
    neighbors_list = tree.query_ball_point(coords, r=radius)
    
    ci_scores = []
    for i, neighbors in enumerate(neighbors_list):
        subject_dbh = dbh[i]
        subject_ci = 0
        for n_idx in neighbors:
            if i == n_idx: continue # Skip self
            
            dist = np.linalg.norm(coords[i] - coords[n_idx])
            if dist > 0:
                n_dbh = dbh[n_idx]
                # Only calculate if the neighbor is alive/valid
                if n_dbh > 0 and subject_dbh > 0:
                    subject_ci += (n_dbh / subject_dbh) / dist
        ci_scores.append(subject_ci)
        
    valid_trees['Competition_Index'] = ci_scores
    
    # Merge the calculated CI back to the main dataframe
    return df.merge(valid_trees[[COL_ID, 'Competition_Index']], on=COL_ID, how='left')

@st.cache_data(show_spinner=True)
def load_and_process_data(csv_path, min_dbh_limit=DEFAULT_MIN_DBH):
    """
    Loads the dataset, filters by Minimum DBH, and generates spatial features.
    """
    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(csv_path, encoding='latin1')

        df.columns = df.columns.str.strip()
        
        # --- CLEAN TEXT COLUMNS ---
        if COL_SPECIES in df.columns: 
            df[COL_SPECIES] = df[COL_SPECIES].astype(str).str.strip()
        if COL_SPECIES_GRP in df.columns: 
            df[COL_SPECIES_GRP] = df[COL_SPECIES_GRP].astype(str).str.strip()
        
        # --- CLEAN NUMERIC COLUMNS ---
        cols_to_clean = [COL_HISTORY, COL_CURRENT, COL_TARGET]
        for c in cols_to_clean:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                # Replace 0 with NaN for calculations, but careful with history logic later
                df[c].replace(0, np.nan, inplace=True)

        # --- FILTERING (Option A) ---
        # Remove small saplings globally based on the Slider Input
        if COL_CURRENT in df.columns:
            df = df[df[COL_CURRENT] >= min_dbh_limit]

        # 1. Feature: Past Growth (History -> Current)
        if COL_HISTORY in df.columns and COL_CURRENT in df.columns:
            # If history is missing, we assume it was smaller or same (0 growth) or handle as NaN
            df['GROWTH_HIST'] = df[COL_CURRENT] - df[COL_HISTORY]

        # 2. Spatial Calculations (Runs ONLY on the filtered valid trees)
        spatial_df = df.dropna(subset=[COL_X, COL_Y, COL_CURRENT]).copy()
        if not spatial_df.empty:
            coords = spatial_df[[COL_X, COL_Y]].values
            tree = cKDTree(coords)
            
            # Nearest Neighbor Distance (k=2 because 1st is itself)
            dists, _ = tree.query(coords, k=2)
            spatial_df['Nearest_Neighbor_Dist'] = dists[:, 1]
            
            # Local Density
            counts = tree.query_ball_point(coords, r=DENSITY_RADIUS, return_length=True)
            spatial_df['Local_Density'] = counts - 1
            
            df = df.merge(spatial_df[[COL_ID, 'Nearest_Neighbor_Dist', 'Local_Density']], on=COL_ID, how='left')

        # 3. Calculate Competition Index
        df = calculate_competition_index(df)

        # 4. Final Cleanup
        # We need these columns to exist for the App to work
        df.dropna(subset=[COL_CURRENT, 'GROWTH_HIST', 'Competition_Index'], inplace=True)
        
        return df

    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None

def run_predictions(df, model_grow, model_mort, encoder):
    """
    Runs both the Growth Regression model and the Mortality Classification model.
    """
    # Encode Species
    if 'SP_Encoded' not in df.columns:
        known_classes = encoder.classes_
        map_dict = {sp: i for i, sp in enumerate(known_classes)}
        df['SP_Encoded'] = df[COL_SPECIES].map(map_dict).fillna(-1) 
    
    # Feature Vector (Must match training!)
    features = [COL_CURRENT, 'GROWTH_HIST', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded']
    
    # Filter valid rows for prediction
    pred_df = df[features].dropna()
    pred_df = pred_df[pred_df['SP_Encoded'] != -1]
    
    if not pred_df.empty:
        # 1. Predict Growth (Size)
        predicted_future = model_grow.predict(pred_df)
        df.loc[pred_df.index, 'Predicted_Size'] = predicted_future 
        df.loc[pred_df.index, 'Predicted_Growth'] = predicted_future - pred_df[COL_CURRENT]
        
        # 2. Predict Mortality Risk (Probability)
        # model.predict_proba returns [Prob_0, Prob_1]. We want Prob_1 (Death)
        if model_mort is not None:
            risk_probs = model_mort.predict_proba(pred_df)[:, 1] 
            df.loc[pred_df.index, 'Mortality_Risk'] = risk_probs
        
    return df