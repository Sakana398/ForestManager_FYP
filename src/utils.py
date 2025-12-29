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

def calculate_competition_index(df, radius=COMPETITION_RADIUS):
    if COL_CURRENT not in df.columns: return df

    valid_trees = df.dropna(subset=[COL_X, COL_Y, COL_CURRENT]).copy().reset_index(drop=True)
    coords = valid_trees[[COL_X, COL_Y]].values
    dbh = valid_trees[COL_CURRENT].values
    
    tree = cKDTree(coords)
    neighbors_list = tree.query_ball_point(coords, r=radius)
    
    ci_scores = []
    for i, neighbors in enumerate(neighbors_list):
        subject_dbh = dbh[i]
        subject_ci = 0
        for n_idx in neighbors:
            if i == n_idx: continue
            dist = np.linalg.norm(coords[i] - coords[n_idx])
            if dist > 0:
                n_dbh = dbh[n_idx]
                if n_dbh > 0 and subject_dbh > 0:
                    subject_ci += (n_dbh / subject_dbh) / dist
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
        
        # Clean Text
        if COL_SPECIES in df.columns: 
            df[COL_SPECIES] = df[COL_SPECIES].astype(str).str.strip()
        if COL_SPECIES_GRP in df.columns: 
            df[COL_SPECIES_GRP] = df[COL_SPECIES_GRP].astype(str).str.strip()
        
        # Clean Numeric Columns (All Years)
        all_year_cols = list(COL_YEARS.values())
        for c in all_year_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                # Do NOT fillna(0) here, keep as NaN to plot gaps correctly
        
        # Filter (Dynamic)
        if COL_CURRENT in df.columns:
            df = df[df[COL_CURRENT] >= min_dbh_limit]

        # Growth History Feature (2000->2005)
        if COL_HISTORY in df.columns and COL_CURRENT in df.columns:
            # For the model, we need a concrete value. Fill missing history with current (0 growth assumption)
            hist_vals = df[COL_HISTORY].fillna(df[COL_CURRENT])
            df['GROWTH_HIST'] = df[COL_CURRENT] - hist_vals

        # Spatial Calculations
        spatial_df = df.dropna(subset=[COL_X, COL_Y, COL_CURRENT]).copy()
        
        if not spatial_df.empty:
            coords = spatial_df[[COL_X, COL_Y]].values
            tree = cKDTree(coords)
            
            dists, _ = tree.query(coords, k=2)
            spatial_df['Nearest_Neighbor_Dist'] = dists[:, 1]
            
            counts = tree.query_ball_point(coords, r=DENSITY_RADIUS, return_length=True)
            spatial_df['Local_Density'] = counts - 1
            
            df = df.merge(spatial_df[[COL_ID, 'Nearest_Neighbor_Dist', 'Local_Density']], on=COL_ID, how='left')

        # Competition Index
        df = calculate_competition_index(df)

        # Final Validation
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
        # 1. Growth
        predicted_future = model_grow.predict(pred_df)
        corrected_future = np.maximum(predicted_future, pred_df[COL_CURRENT]) # Prevent shrinking
        
        df.loc[pred_df.index, 'Predicted_Size'] = corrected_future 
        df.loc[pred_df.index, 'Predicted_Growth'] = corrected_future - pred_df[COL_CURRENT]
        
        # 2. Mortality
        if model_mort is not None:
            risk_probs = model_mort.predict_proba(pred_df)[:, 1] 
            df.loc[pred_df.index, 'Mortality_Risk'] = risk_probs
        else:
            df['Mortality_Risk'] = 0.0
        
    return df